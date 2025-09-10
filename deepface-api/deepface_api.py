from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import numpy as np
from PIL import Image
import io
import cv2
import logging
import os
import threading
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# EDA 信号处理相关导入
try:
    import neurokit2 as nk
    import pandas as pd
    from scipy import signal
    NEUROKIT_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("NeuroKit2 可用")
except ImportError as e:
    NEUROKIT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"NeuroKit2 不可用: {e}")

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 性能优化配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # 强制使用CPU，避免GPU初始化开销

# VA映射表：从7种情绪标签到VA值的映射
VA_MAPPING = {
    'angry': {'valence': -0.51, 'arousal': 0.59},
    'disgust': {'valence': -0.60, 'arousal': 0.35},
    'fear': {'valence': -0.64, 'arousal': 0.60},
    'happy': {'valence': 0.81, 'arousal': 0.51},
    'sad': {'valence': -0.63, 'arousal': -0.27},
    'surprise': {'valence': 0.40, 'arousal': 0.67},
    'neutral': {'valence': 0.00, 'arousal': 0.00}
}

# 全局变量用于模型预热
_model_initialized = False
_init_lock = threading.Lock()

# 批量处理任务存储
batch_jobs = {}  # {job_id: {status, progress, result, timestamp}}

def initialize_model():
    """预热模型，避免首次调用的延迟"""
    global _model_initialized
    if not _model_initialized:
        with _init_lock:
            if not _model_initialized:  # 双重检查
                try:
                    logger.info("正在预热DeepFace模型...")
                    # 创建一个小的测试图像来预热模型
                    test_img = np.ones((48, 48, 3), dtype=np.uint8) * 128
                    DeepFace.analyze(
                        img_path=test_img,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    _model_initialized = True
                    logger.info("DeepFace模型预热完成")
                except Exception as e:
                    logger.warning(f"模型预热失败: {e}")

def calculate_va_from_emotions(emotion_probs):
    """根据情绪概率分布计算VA值（优化版本）"""
    valence = 0.0
    arousal = 0.0
    
    for emotion, prob in emotion_probs.items():
        if emotion in VA_MAPPING:
            # 将概率值作为权重，确保转换为Python float类型
            weight = float(prob) * 0.01  # DeepFace返回0-100的概率值，转换为0-1
            valence += VA_MAPPING[emotion]['valence'] * weight
            arousal += VA_MAPPING[emotion]['arousal'] * weight
    
    return float(valence), float(arousal)  # 确保返回Python float类型

# ========================= EDA 信号预处理模块 =========================

class EDAPreprocessor:
    """
    EDA信号预处理器 - 使用官方NeuroKit2库
    """
    
    def __init__(self):
        self.available_methods = {
            'neurokit': 'NeuroKit2官方方法',
            'biosppy': 'BioSPPy兼容方法',
            'cvxeda': 'cvxEDA分解方法',
            'none': '无预处理'
        }
    
    def clean_eda(self, eda_signal, sampling_rate=50.0, method='neurokit'):
        """
        使用NeuroKit2清理EDA信号
        
        Args:
            eda_signal (list/array): 原始EDA信号
            sampling_rate (float): 采样率 (Hz)
            method (str): 预处理方法
            
        Returns:
            dict: 包含清理后的信号和相关信息
        """
        if not NEUROKIT_AVAILABLE:
            logger.warning("NeuroKit2不可用，使用简化预处理")
            return self._fallback_clean(eda_signal, sampling_rate)
        
        try:
            # 转换为numpy数组
            signal_array = np.array(eda_signal, dtype=float)
            
            if method == 'none':
                return {
                    'cleaned_signal': signal_array.tolist(),
                    'method_used': 'none',
                    'sampling_rate': sampling_rate,
                    'original_length': len(signal_array),
                    'processed_length': len(signal_array),
                    'quality_score': 1.0
                }
            
            # 使用NeuroKit2处理EDA信号
            if method == 'neurokit':
                # NeuroKit2的标准EDA清理方法
                cleaned = nk.eda_clean(signal_array, sampling_rate=sampling_rate, method='neurokit')
            elif method == 'biosppy':
                # BioSPPy兼容的方法
                cleaned = nk.eda_clean(signal_array, sampling_rate=sampling_rate, method='biosppy')
            elif method == 'cvxeda':
                # cvxEDA方法（如果可用）
                try:
                    cleaned = nk.eda_clean(signal_array, sampling_rate=sampling_rate, method='cvxeda')
                except:
                    logger.warning("cvxEDA方法不可用，回退到neurokit方法")
                    cleaned = nk.eda_clean(signal_array, sampling_rate=sampling_rate, method='neurokit')
            else:
                cleaned = nk.eda_clean(signal_array, sampling_rate=sampling_rate, method='neurokit')
            
            # 计算信号质量评估
            quality_score = self._assess_signal_quality(signal_array, cleaned)
            
            result = {
                'cleaned_signal': cleaned.tolist(),
                'method_used': method,
                'sampling_rate': sampling_rate,
                'original_length': len(signal_array),
                'processed_length': len(cleaned),
                'quality_score': quality_score,
                'preprocessing_info': {
                    'neurokit_version': nk.__version__ if hasattr(nk, '__version__') else 'unknown',
                    'filters_applied': f'{method}_method'
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"NeuroKit2 EDA预处理失败: {e}")
            return self._fallback_clean(eda_signal, sampling_rate)
    
    def decompose_eda(self, eda_signal, sampling_rate=50.0):
        """
        分解EDA信号为慢性和快性成分
        """
        if not NEUROKIT_AVAILABLE:
            return self._fallback_decompose(eda_signal)
        
        try:
            signal_array = np.array(eda_signal, dtype=float)
            
            # 使用NeuroKit2分解EDA信号
            signals, info = nk.eda_process(signal_array, sampling_rate=sampling_rate)
            
            result = {
                'tonic': signals['EDA_Tonic'].tolist(),  # 慢性成分 (SCL)
                'phasic': signals['EDA_Phasic'].tolist(), # 快性成分 (SCR)
                'clean': signals['EDA_Clean'].tolist(),   # 清理后的信号
                'sampling_rate': sampling_rate,
                'peaks': info.get('SCR_Peaks', []).tolist() if 'SCR_Peaks' in info else [],
                'decomposition_method': 'neurokit2'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"EDA信号分解失败: {e}")
            return self._fallback_decompose(eda_signal)
    
    def _assess_signal_quality(self, original, cleaned):
        """评估信号质量"""
        try:
            # 计算信噪比改善
            original_std = np.std(original)
            cleaned_std = np.std(cleaned)
            
            if original_std == 0:
                return 1.0
            
            # 简单的质量评分 (0-1)
            snr_improvement = min(original_std / (cleaned_std + 1e-6), 2.0)
            quality_score = min(snr_improvement / 2.0, 1.0)
            
            return float(quality_score)
        except:
            return 0.5
    
    def _fallback_clean(self, eda_signal, sampling_rate):
        """当NeuroKit2不可用时的备用清理方法"""
        signal_array = np.array(eda_signal, dtype=float)
        
        # 简单的移动平均滤波
        window_size = max(3, int(sampling_rate * 0.1))  # 0.1秒的窗口
        if len(signal_array) < window_size:
            cleaned = signal_array
        else:
            cleaned = np.convolve(signal_array, np.ones(window_size)/window_size, mode='same')
        
        return {
            'cleaned_signal': cleaned.tolist(),
            'method_used': 'fallback_moving_average',
            'sampling_rate': sampling_rate,
            'original_length': len(signal_array),
            'processed_length': len(cleaned),
            'quality_score': 0.5,
            'warning': 'NeuroKit2不可用，使用简化预处理'
        }
    
    def _fallback_decompose(self, eda_signal):
        """备用分解方法"""
        signal_array = np.array(eda_signal, dtype=float)
        
        # 简单的高通/低通滤波分解
        try:
            from scipy.signal import butter, filtfilt
            
            # 低通滤波获得慢性成分 (< 0.05 Hz)
            b_low, a_low = butter(2, 0.05, btype='low', fs=50)
            tonic = filtfilt(b_low, a_low, signal_array)
            
            # 快性成分 = 原信号 - 慢性成分
            phasic = signal_array - tonic
            
            return {
                'tonic': tonic.tolist(),
                'phasic': phasic.tolist(),
                'clean': signal_array.tolist(),
                'sampling_rate': 50.0,
                'peaks': [],
                'decomposition_method': 'fallback_butterworth',
                'warning': 'NeuroKit2不可用，使用简化分解'
            }
        except:
            # 最简单的分解
            mean_val = np.mean(signal_array)
            tonic = np.full_like(signal_array, mean_val)
            phasic = signal_array - tonic
            
            return {
                'tonic': tonic.tolist(),
                'phasic': phasic.tolist(),
                'clean': signal_array.tolist(),
                'sampling_rate': 50.0,
                'peaks': [],
                'decomposition_method': 'fallback_mean',
                'warning': 'NeuroKit2和SciPy不可用，使用最简分解'
            }

# 创建EDA预处理器实例
eda_processor = EDAPreprocessor()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 确保模型已初始化
        initialize_model()
        
        data = request.get_json()
        
        # 获取base64图像数据
        img_data = data.get('img', '')
        if img_data.startswith('data:image'):
            img_data = img_data.split(',')[1]
        
        # 优化图像处理
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # 如果图像太大，先缩放以提高性能
        if max(img.size) > 640:
            img.thumbnail((640, 640), Image.Resampling.LANCZOS)
        
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 使用DeepFace分析情绪
        # 优化：使用较小的模型提高速度
        result = DeepFace.analyze(
            img_path=img_array,
            actions=['emotion'],
            enforce_detection=False,  # 允许低置信度的检测
            detector_backend='opencv',  # 使用较快的检测器
            silent=True
        )
        
        # 处理结果格式（DeepFace可能返回列表或字典）
        if isinstance(result, list):
            if len(result) == 0:
                return jsonify({
                    'status': 'error',
                    'message': '未检测到面部',
                    'face': False
                })
            analysis_result = result[0]
        else:
            analysis_result = result
            
        if not analysis_result or 'emotion' not in analysis_result:
            return jsonify({
                'status': 'error',
                'message': '未检测到面部或情绪分析失败',
                'face': False
            })
        
        # 获取情绪分析结果
        emotion_probs = analysis_result['emotion']
        dominant_emotion = analysis_result['dominant_emotion']
        
        # 转换emotion_probs中的numpy类型为Python类型，避免JSON序列化错误
        emotion_probs_clean = {k: float(v) for k, v in emotion_probs.items()}
        
        # 计算VA值
        valence, arousal = calculate_va_from_emotions(emotion_probs_clean)
        
        logger.info(f"检测到情绪: {dominant_emotion}, VA值: V={valence:.3f}, A={arousal:.3f}")
        
        # 返回优化的响应格式
        response_data = {
            'status': 'success',
            'face': True,
            'valence_arousal': {
                'valence': valence,
                'arousal': arousal
            },
            'emotions': emotion_probs_clean,
            'dominant_emotion': dominant_emotion
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'分析失败: {str(e)}',
            'face': False
        }), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """批量分析图像 - 用于优化性能"""
    try:
        # 确保模型已初始化
        initialize_model()
        
        data = request.get_json()
        images_data = data.get('images', [])
        
        if not images_data:
            return jsonify({
                'status': 'error',
                'message': '没有提供图像数据'
            }), 400
        
        results = []
        
        # 使用ThreadPoolExecutor进行并行处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 准备所有图像
            processed_images = []
            for i, img_data in enumerate(images_data):
                try:
                    if img_data.startswith('data:image'):
                        img_data = img_data.split(',')[1]
                    
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # 缩放图像
                    if max(img.size) > 640:
                        img.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    
                    img_array = np.array(img)
                    processed_images.append((i, img_array))
                    
                except Exception as e:
                    logger.error(f"图像 {i} 预处理失败: {str(e)}")
                    results.append({
                        'index': i,
                        'status': 'error',
                        'message': f'图像预处理失败: {str(e)}',
                        'face': False
                    })
            
            # 并行分析
            def analyze_single(img_data):
                index, img_array = img_data
                try:
                    result = DeepFace.analyze(
                        img_path=img_array,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True
                    )
                    
                    # 处理结果
                    if isinstance(result, list):
                        if len(result) == 0:
                            return {
                                'index': index,
                                'status': 'error',
                                'message': '未检测到面部',
                                'face': False
                            }
                        analysis_result = result[0]
                    else:
                        analysis_result = result
                    
                    if not analysis_result or 'emotion' not in analysis_result:
                        return {
                            'index': index,
                            'status': 'error',
                            'message': '情绪分析失败',
                            'face': False
                        }
                    
                    emotion_probs = analysis_result['emotion']
                    dominant_emotion = analysis_result['dominant_emotion']
                    emotion_probs_clean = {k: float(v) for k, v in emotion_probs.items()}
                    valence, arousal = calculate_va_from_emotions(emotion_probs_clean)
                    
                    return {
                        'index': index,
                        'status': 'success',
                        'face': True,
                        'valence_arousal': {
                            'valence': valence,
                            'arousal': arousal
                        },
                        'emotions': emotion_probs_clean,
                        'dominant_emotion': dominant_emotion
                    }
                    
                except Exception as e:
                    logger.error(f"批量分析图像 {index} 失败: {str(e)}")
                    return {
                        'index': index,
                        'status': 'error',
                        'message': f'分析失败: {str(e)}',
                        'face': False
                    }
            
            # 执行并行分析
            future_to_result = {executor.submit(analyze_single, img_data): img_data for img_data in processed_images}
            
            for future in as_completed(future_to_result):
                result = future.result()
                results.append(result)
        
        # 按索引排序结果
        results.sort(key=lambda x: x['index'])
        
        logger.info(f"批量分析完成: {len(results)} 个图像")
        
        return jsonify({
            'status': 'success',
            'total': len(images_data),
            'processed': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"批量分析失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'批量分析失败: {str(e)}'
        }), 500
        
        img_array = np.array(img)
        
        # 转换为OpenCV格式
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        logger.info(f"分析图像，尺寸: {img_array.shape}")
        
        # DeepFace分析（使用更快的设置）
        result = DeepFace.analyze(
            img_path=img_array, 
            actions=['emotion'],
            enforce_detection=False,  # 不强制检测，提高速度
            detector_backend='opencv'  # 使用最快的检测器
        )
        
        # 处理结果
        if isinstance(result, list):
            analysis_result = result[0] if result else None
        else:
            analysis_result = result
            
        if not analysis_result or 'emotion' not in analysis_result:
            return jsonify({
                'status': 'error',
                'message': '未检测到面部或情绪分析失败',
                'face': False
            })
        
        # 获取情绪分析结果
        emotion_probs = analysis_result['emotion']
        dominant_emotion = analysis_result['dominant_emotion']
        
        # 转换emotion_probs中的numpy类型为Python类型，避免JSON序列化错误
        emotion_probs_clean = {k: float(v) for k, v in emotion_probs.items()}
        
        # 计算VA值
        valence, arousal = calculate_va_from_emotions(emotion_probs_clean)
        
        logger.info(f"检测到情绪: {dominant_emotion}, VA值: V={valence:.3f}, A={arousal:.3f}")
        
        # 返回优化的响应格式
        response_data = {
            'status': 'success',
            'face': True,
            'valence_arousal': {
                'valence': valence,
                'arousal': arousal
            },
            'emotions': emotion_probs_clean,
            'dominant_emotion': str(dominant_emotion)  # 确保是字符串类型
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"分析错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'face': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'message': 'DeepFace API is running',
        'version': '1.0.0',
        'supported_actions': ['emotion', 'age', 'gender', 'race']
    })

@app.route('/test', methods=['GET'])
def test():
    """测试端点，返回服务器信息"""
    return jsonify({
        'message': 'DeepFace API 测试成功',
        'version': '批量处理版本',
        'server': 'Flask',
        'deepface': 'Ready',
        'endpoints': {
            '/health': 'GET - 健康检查',
            '/analyze': 'POST - 图像分析',
            '/batch_upload': 'POST - 批量上传分析',
            '/batch_status/<job_id>': 'GET - 查询任务状态',
            '/batch_download/<job_id>': 'GET - 下载分析结果',
            '/test': 'GET - 测试端点'
        }
    })

def process_single_frame(frame_data):
    """处理单帧数据"""
    try:
        frame_number, img_array, timestamp = frame_data
        
        # 调整图像大小以提高速度
        height, width = img_array.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            new_height, new_width = int(height * scale), int(width * scale)
            img_array = cv2.resize(img_array, (new_width, new_height))
        
        # DeepFace 分析
        result = DeepFace.analyze(
            img_path=img_array,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if isinstance(result, list):
            result = result[0]
        
        if result and 'emotion' in result:
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            
            # 转换为Python类型避免JSON序列化错误
            emotions_clean = {k: float(v) for k, v in emotions.items()}
            
            # 计算 VA 值
            valence, arousal = calculate_va_from_emotions(emotions_clean)
            
            return {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'valence': valence,
                'arousal': arousal,
                'emotions': emotions_clean,
                'dominant_emotion': str(dominant_emotion),
                'face': True
            }
        else:
            return {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'face': False
            }
            
    except Exception as e:
        logger.error(f"处理帧 {frame_number} 失败: {e}")
        return {
            'frame_number': frame_data[0],
            'timestamp': frame_data[2],
            'face': False,
            'error': str(e)
        }

def process_batch_job(job_id, frames_data):
    """后台处理批量任务"""
    try:
        logger.info(f"开始批量处理任务 {job_id}: {len(frames_data)} 帧")
        
        # 更新状态
        batch_jobs[job_id]['status'] = 'processing'
        batch_jobs[job_id]['progress'] = 0
        
        # 并行处理帧（使用4个线程，安全起见）
        max_workers = min(4, len(frames_data))
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {
                executor.submit(process_single_frame, frame_data): frame_data[0]
                for frame_data in frames_data
            }
            
            completed = 0
            for future in as_completed(future_to_frame):
                frame_number = future_to_frame[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # 更新进度
                    progress = (completed / len(frames_data)) * 100
                    batch_jobs[job_id]['progress'] = progress
                    
                    if completed % 5 == 0:
                        logger.info(f"任务 {job_id}: 已处理 {completed}/{len(frames_data)} 帧")
                        
                except Exception as e:
                    logger.error(f"帧 {frame_number} 处理异常: {e}")
                    results.append({
                        'frame_number': frame_number,
                        'face': False,
                        'error': str(e)
                    })
                    completed += 1
                    batch_jobs[job_id]['progress'] = (completed / len(frames_data)) * 100
        
        # 按帧序号排序
        results.sort(key=lambda x: x.get('frame_number', 0))
        
        # 统计结果
        face_count = sum(1 for r in results if r.get('face', False))
        
        # 更新任务状态
        batch_jobs[job_id]['status'] = 'completed'
        batch_jobs[job_id]['progress'] = 100
        batch_jobs[job_id]['result'] = {
            'results': results,
            'stats': {
                'total_frames': len(results),
                'face_detected_frames': face_count,
                'processing_time': time.time() - batch_jobs[job_id]['start_time']
            }
        }
        
        logger.info(f"任务 {job_id} 完成: {len(results)} 帧, {face_count} 帧检测到人脸")
        
    except Exception as e:
        logger.error(f"批量处理任务 {job_id} 失败: {e}")
        batch_jobs[job_id]['status'] = 'failed'
        batch_jobs[job_id]['error'] = str(e)

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """批量上传帧数据，开始后台处理"""
    try:
        # 确保模型已初始化
        initialize_model()
        
        data = request.get_json()
        frames = data.get('frames', [])
        
        if not frames:
            return jsonify({'error': '没有提供帧数据'}), 400
        
        # 生成任务ID
        job_id = str(uuid.uuid4())
        
        # 创建任务记录
        batch_jobs[job_id] = {
            'status': 'uploading',
            'progress': 0,
            'start_time': time.time(),
            'timestamp': time.time()
        }
        
        logger.info(f"创建批量处理任务 {job_id}: {len(frames)} 帧")
        
        # 准备帧数据
        frames_data = []
        for i, frame_info in enumerate(frames):
            try:
                # 解码 base64 图像
                img_data = frame_info['image']
                if img_data.startswith('data:image'):
                    img_data = img_data.split(',')[1]
                
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img)
                
                # 转换为 OpenCV 格式
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                frames_data.append((
                    frame_info.get('frame_number', i),
                    img_array,
                    frame_info.get('timestamp', 0)
                ))
                
            except Exception as e:
                logger.error(f"解码帧 {i} 失败: {e}")
                continue
        
        # 启动后台处理线程
        thread = threading.Thread(
            target=process_batch_job, 
            args=(job_id, frames_data)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'message': f'批量处理任务已启动，共 {len(frames_data)} 帧'
        })
        
    except Exception as e:
        logger.error(f"批量上传失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_status/<job_id>', methods=['GET'])
def batch_status(job_id):
    """查询批量处理状态"""
    if job_id not in batch_jobs:
        return jsonify({'error': '任务不存在'}), 404
    
    job = batch_jobs[job_id]
    
    response = {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress']
    }
    
    if job['status'] == 'failed':
        response['error'] = job.get('error', '未知错误')
    elif job['status'] == 'completed':
        response['stats'] = job['result']['stats']
    
    return jsonify(response)

@app.route('/batch_download/<job_id>', methods=['GET'])
def batch_download(job_id):
    """下载批量处理结果"""
    if job_id not in batch_jobs:
        return jsonify({'error': '任务不存在'}), 404
    
    job = batch_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': f'任务状态: {job["status"]}'}), 400
    
    result = job['result']
    
    return jsonify({
        'status': 'success',
        'job_id': job_id,
        'results': result['results'],
        'stats': result['stats']
    })

# ========================= EDA 预处理 API 端点 =========================

@app.route('/eda/clean', methods=['POST'])
def eda_clean():
    """EDA信号清理API"""
    try:
        data = request.get_json()
        
        # 获取必需参数
        eda_signal = data.get('signal', [])
        if not eda_signal:
            return jsonify({'error': '缺少signal参数'}), 400
        
        # 获取可选参数
        sampling_rate = data.get('sampling_rate', 50.0)
        method = data.get('method', 'neurokit')
        
        # 验证方法
        if method not in eda_processor.available_methods:
            return jsonify({
                'error': f'不支持的方法: {method}',
                'available_methods': list(eda_processor.available_methods.keys())
            }), 400
        
        # 执行清理
        result = eda_processor.clean_eda(
            eda_signal=eda_signal,
            sampling_rate=sampling_rate,
            method=method
        )
        
        return jsonify({
            'status': 'success',
            'data': result,
            'neurokit_available': NEUROKIT_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"EDA清理失败: {e}")
        return jsonify({'error': f'EDA清理失败: {str(e)}'}), 500

@app.route('/eda/decompose', methods=['POST'])
def eda_decompose():
    """EDA信号分解API - 分离慢性和快性成分"""
    try:
        data = request.get_json()
        
        # 获取必需参数
        eda_signal = data.get('signal', [])
        if not eda_signal:
            return jsonify({'error': '缺少signal参数'}), 400
        
        # 获取可选参数
        sampling_rate = data.get('sampling_rate', 50.0)
        
        # 执行分解
        result = eda_processor.decompose_eda(
            eda_signal=eda_signal,
            sampling_rate=sampling_rate
        )
        
        return jsonify({
            'status': 'success',
            'data': result,
            'neurokit_available': NEUROKIT_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"EDA分解失败: {e}")
        return jsonify({'error': f'EDA分解失败: {str(e)}'}), 500

@app.route('/eda/info', methods=['GET'])
def eda_info():
    """获取EDA预处理功能信息"""
    return jsonify({
        'status': 'success',
        'neurokit_available': NEUROKIT_AVAILABLE,
        'available_methods': eda_processor.available_methods,
        'endpoints': {
            '/eda/clean': 'EDA信号清理',
            '/eda/decompose': 'EDA信号分解(慢性/快性成分)',
            '/eda/info': '获取EDA功能信息'
        },
        'example_request': {
            'clean': {
                'signal': '[0.1, 0.2, 0.15, ...]',
                'sampling_rate': 50.0,
                'method': 'neurokit'
            },
            'decompose': {
                'signal': '[0.1, 0.2, 0.15, ...]',
                'sampling_rate': 50.0
            }
        }
    })

if __name__ == '__main__':
    logger.info("启动 DeepFace API 服务器...")
    logger.info("服务器将在 http://localhost:5000 启动")
    logger.info("Health check: http://localhost:5000/health")
    logger.info("Test endpoint: http://localhost:5000/test")
    logger.info("Analyze endpoint: http://localhost:5000/analyze")
    
    # 在启动时预热模型
    initialize_model()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
