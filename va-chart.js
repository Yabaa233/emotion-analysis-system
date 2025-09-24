// Valence / Arousal 图（Chart.js v2）
// 保持与原 EmoChart 相同的公开接口：constructor(el), visible, reset(), update(), updateNoData()
// 支持显示：Valence (绿), Arousal (黄) 两条曲线
class VAChart {
  constructor(el, initialMax = 300, step = 2) {
    this.MAX = initialMax; // 初始 x 轴窗口宽度（秒）
    this.STEP = step;      // x 轴刻度
    this.chart = this._chart;

    this._element = el;
    const ctx = el.getContext('2d');

    // 两条曲线 + "未检测到人脸"遮罩 + 播放标记 + Attention（白线）
    const datasets = [
      {
        label: 'Valence (-1 ~ 1)',
        fill: false,
        borderWidth: 2,
        borderColor: '#78ff8e',
        data: [],
        lineTension: 0,
        pointRadius: 0,
        showLine: true
      },
      {
        label: 'Arousal (-1 ~ 1)',
        fill: false,
        borderWidth: 2,
        borderColor: '#ffe063',
        data: [],
        lineTension: 0,
        pointRadius: 0,
        showLine: true
      },
      {
        label: 'Face not detected',
        fill: true,
        backgroundColor: 'rgba(170,170,170,0.6)',
        borderColor: 'rgba(233, 233, 233, 0.6)',
        data: [],
        lineTension: 0,
        pointRadius: 0
      },
      {
        label: 'Playback Marker',
        fill: false,
        borderWidth: 2,
        borderColor: '#ff0000',
        data: [],
        lineTension: 0,
        pointRadius: 5
      },
      {
        label: 'Valence Rate (dV/dt)',
        fill: false,
        borderWidth: 1,
        borderColor: '#00ff88',
        borderDash: [5, 5],
        data: [],
        lineTension: 0,
        pointRadius: 0,
        showLine: true,
        hidden: true // 默认隐藏
      },
      {
        label: 'Arousal Rate (dA/dt)',
        fill: false,
        borderWidth: 1,
        borderColor: '#ffaa00',
        borderDash: [5, 5],
        data: [],
        lineTension: 0,
        pointRadius: 0,
        showLine: true,
        hidden: true // 默认隐藏
      },
      {
        label: 'Rise Events',
        fill: false,
        borderWidth: 2,
        backgroundColor: '#4caf50',
        borderColor: '#2e7d32',
        data: [],
        lineTension: 0,
        pointRadius: 10,
        pointStyle: 'triangle',
        pointHoverRadius: 12,
        showLine: false,
        hidden: true // 默认隐藏
      },
      {
        label: 'Fall Events',
        fill: false,
        borderWidth: 2,
        backgroundColor: '#f44336',
        borderColor: '#c62828',
        data: [],
        lineTension: 0,
        pointRadius: 10,
        pointStyle: 'rectRot',
        pointHoverRadius: 12,
        showLine: false,
        hidden: true // 默认隐藏
      },
      {
        label: 'GSR Conductance (μS)',
        fill: false,
        borderWidth: 2,
        borderColor: '#ff9800',
        data: [],
        lineTension: 0,
        pointRadius: 0,
        showLine: true,
        yAxisID: 'y-axis-gsr',
        hidden: true // 默认隐藏
      },
      {
        label: 'PPG Heart Rate (mV)',
        fill: false,
        borderWidth: 2,
        borderColor: '#e91e63',
        data: [],
        lineTension: 0,
        pointRadius: 0,
        showLine: true,
        yAxisID: 'y-axis-ppg',
        hidden: true // 默认隐藏
      },
      {
        label: 'Manual Anomaly Marks',
        fill: false,
        borderWidth: 3,
        backgroundColor: '#ff4444',
        borderColor: '#cc0000',
        data: [],
        lineTension: 0,
        pointRadius: 8,
        pointStyle: 'star',
        pointHoverRadius: 12,
        showLine: false,
        hidden: false // 默认显示
      }
    ];

    const config = {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 4,
        animation: { duration: 0 },
        legend: { 
          position: 'top', 
          labels: { fontColor: '#fff' },
          onClick: (e, legendItem) => {
            const index = legendItem.datasetIndex;
            const meta = this._chart.getDatasetMeta(index);
            meta.hidden = meta.hidden === null ? !datasets[index].hidden : !meta.hidden;
            this._chart.update();
          }
        },
        scales: {
          xAxes: [{
            type: 'linear',
            display: true,
            scaleLabel: { display: true, labelString: 'time (sec)' },
            ticks: { min: 0, suggestedMax: this.MAX + this.STEP, stepSize: this.STEP }
          }],
          yAxes: [{
            id: 'y-axis-main',
            position: 'left',
            display: true,
            scaleLabel: { display: true, labelString: 'Valence / Arousal' },
            ticks: { min: -1, max: 1, stepSize: 0.5, fontColor: '#fff' },
            gridLines: {
              color: 'rgba(255,255,255,0.08)',
              zeroLineColor: 'rgba(255,255,255,0.35)',
              zeroLineWidth: 2
            }
          }, {
            id: 'y-axis-gsr',
            position: 'right',
            display: false,
            scaleLabel: { display: true, labelString: 'GSR (μS)', fontColor: '#ff9800' },
            ticks: { fontColor: '#ff9800' },
            gridLines: { display: false }
          }, {
            id: 'y-axis-ppg',
            position: 'right',
            display: false,
            scaleLabel: { display: true, labelString: 'PPG (mV)', fontColor: '#e91e63' },
            ticks: { fontColor: '#e91e63' },
            gridLines: { display: false }
          }]
        },
        tooltips: { mode: 'index', intersect: false }
      }
    };

    Chart.defaults.global.defaultFontColor = "#fff";

    this._chart = new Chart(ctx, config);
    this._datasets = datasets;   // 0: Valence, 1: Arousal, 2: No-face, 3: Marker, 4: Valence_dt, 5: Arousal_dt, 6: Rise Events, 7: Fall Events, 8: GSR, 9: PPG
    this._config = config;

    this._noDataTime = null;
    this._lastUpdateTime = null;
  }

  // 调整可视窗口秒数
  setZoom(seconds) {
    this.MAX = seconds;
    const currentMin = this._config.options.scales.xAxes[0].ticks.min || 0;
    this._config.options.scales.xAxes[0].ticks.max = currentMin + this.MAX + this.STEP;
    this._chart.update();
  }

  set visible(visible) { this._element.style.display = visible ? 'block' : 'none'; }

  reset() {
    this._datasets.forEach(d => d.data = []);
    const min = 0;
    const max = this.MAX + this.STEP;
    this._config.options.scales.xAxes[0].ticks.min = min;
    this._config.options.scales.xAxes[0].ticks.max = max;
    this._noDataTime = null;
    this._lastUpdateTime = null;
    this._chart.update();
  }

  // 修复时间轴跳转逻辑
  setWindowByTime(t) {
    const span = this.MAX + this.STEP;
    const base = Math.floor(t / span) * span;
    if (this._config.options.scales.xAxes[0].ticks.min !== base) {
      this._config.options.scales.xAxes[0].ticks.min = base;
      this._config.options.scales.xAxes[0].ticks.max = base + span;
      this._chart.update();
    }
  }

  // 修复横轴范围动态更新逻辑
  _updateMinMax(time) {
    const max = this._config.options.scales.xAxes[0].ticks.max || this.MAX;
    const min = this._config.options.scales.xAxes[0].ticks.min || 0;
    if (time >= max || time < min) {
      const span = this.MAX + this.STEP;
      const base = Math.floor(time / span) * span;
      this._config.options.scales.xAxes[0].ticks.min = base;
      this._config.options.scales.xAxes[0].ticks.max = base + span;
      this._chart.update();
    }
  }

  update(time, data) {
    const v = clamp(data.valence, -1, 1);
    const a = clamp(data.arousal, -1, 1);

    // 清理超出范围的数据点
    const min = this._config.options.scales.xAxes[0].ticks.min || 0;
    const max = this._config.options.scales.xAxes[0].ticks.max || this.MAX;
    this._datasets.forEach(dataset => {
      dataset.data = dataset.data.filter(point => point.x >= min && point.x <= max);
    });

    this._datasets[0].data.push({ x: time, y: v });
    this._datasets[1].data.push({ x: time, y: a });

    if (this._noDataTime !== null) {
      this._datasets[2].data.push({ x: time, y: 1 });
      this._datasets[2].data.push({ x: time, y: undefined });
      this._noDataTime = null;
    }

    this._updateMinMax(time);
    this._chart.update();
    this._lastUpdateTime = time;
  }

  // 仅更新播放标记点；轻量刷新以呈现标记位置，不触发数据重算
  updatePlaybackMarker(time) {
    if (!this._chart || !this._datasets[3]) return;
    this._datasets[3].data = [{ x: time, y: 0 }];
    this._chart.update(0);
  }

  // 确保显示完整数据
  setViewRange(min, max) {
    this._config.options.scales.xAxes[0].ticks.min = Math.max(0, min);
    this._config.options.scales.xAxes[0].ticks.max = max;
    this._chart.update();
    
    // 同步到情绪图表（如果存在）
    if (typeof window !== 'undefined' && window.emotionChart && window.emotionDetectionEnabled) {
      window.emotionChart.options.scales.xAxes[0].ticks.min = Math.max(0, min);
      window.emotionChart.options.scales.xAxes[0].ticks.max = max;
      window.emotionChart.update();
    }
  }

  // 设置数据集可见性
  setDatasetVisibility(datasetIndex, visible) {
    if (this._datasets[datasetIndex]) {
      this._datasets[datasetIndex].hidden = !visible;
      this._chart.update();
    }
  }

  // 获取数据集可见性
  getDatasetVisibility(datasetIndex) {
    return this._datasets[datasetIndex] ? !this._datasets[datasetIndex].hidden : true;
  }

  resetViewToCurrentTime(currentTime) {
    const span = this.MAX + this.STEP;
    const base = Math.floor(currentTime / span) * span;
    this.setViewRange(base, base + span);
  }

  // 修复"未检测到人脸"逻辑
  updateNoData(time) {
    if (this._lastUpdateTime !== null) {
      this._datasets[2].data.push({ x: this._lastUpdateTime, y: 1 });
      this._lastUpdateTime = null;
    }
    this._datasets[0].data.push({ x: time, y: undefined });
    this._datasets[1].data.push({ x: time, y: undefined });
    this._datasets[2].data.push({ x: time, y: 1 });

    this._updateMinMax(time);
    this._chart.update();
    this._noDataTime = time;
  }

  // 优化鼠标事件处理
  getTimeAtEvent(evt) {
    if (!this._chart || !this._chart.chartArea) return null;
    const rect = this._chart.canvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    const area = this._chart.chartArea;
    if (x < area.left || x > area.right || y < area.top || y > area.bottom) return null;
    const xScale = this._chart.scales && (this._chart.scales['x-axis-0'] || this._chart.scales['x']);
    if (!xScale || typeof xScale.getValueForPixel !== 'function') return null;
    return xScale.getValueForPixel(x);
  }

  // 更新导数数据
  updateDerivativeData(derivativeData) {
    if (!this._datasets[4] || !this._datasets[5]) return;
    
    const valenceRateData = [];
    const arousalRateData = [];
    
    for (const r of derivativeData) {
      if (r.face && Number.isFinite(r.valence_dt)) {
        valenceRateData.push({ x: r.time, y: r.valence_dt });
      }
      if (r.face && Number.isFinite(r.arousal_dt)) {
        arousalRateData.push({ x: r.time, y: r.arousal_dt });
      }
    }
    
    this._datasets[4].data = valenceRateData;
    this._datasets[5].data = arousalRateData;
    this._chart.update();
  }

  // 切换导数显示
  toggleDerivativeDisplay(show) {
    if (this._datasets[4] && this._datasets[5]) {
      const meta4 = this._chart.getDatasetMeta(4);
      const meta5 = this._chart.getDatasetMeta(5);
      meta4.hidden = !show;
      meta5.hidden = !show;
      this._chart.update();
    }
  }

  // 更新事件标记数据
  updateEventData(eventsData, processedData) {
    if (!this._datasets[6] || !this._datasets[7]) return;
    
    const riseEvents = [];
    const fallEvents = [];
    
    for (const event of eventsData) {
      // 找到事件时间点对应的数据，用于确定Y坐标
      const eventRecord = processedData.find(r => Math.abs(r.time - event.t) < 0.1);
      
      let yPos = 0.8; // 默认位置
      if (eventRecord) {
        // 根据事件类型确定显示位置
        if (event.type.includes('valence')) {
          yPos = eventRecord.valence || 0;
        } else if (event.type.includes('arousal')) {
          yPos = eventRecord.arousal || 0;
        }
      }
      
      const eventPoint = {
        x: event.t,
        y: yPos
      };
      
      if (event.type.includes('rise')) {
        riseEvents.push(eventPoint);
      } else if (event.type.includes('fall')) {
        fallEvents.push(eventPoint);
      }
    }
    
    this._datasets[6].data = riseEvents;
    this._datasets[7].data = fallEvents;
    this._chart.update();
  }

  // 切换事件显示
  toggleEventDisplay(show) {
    if (this._datasets[6] && this._datasets[7]) {
      const meta6 = this._chart.getDatasetMeta(6);
      const meta7 = this._chart.getDatasetMeta(7);
      meta6.hidden = !show;
      meta7.hidden = !show;
      this._chart.update();
    }
  }

  // 更新GSR数据
  updateGSRData(gsrData) {
    if (!this._datasets[8]) {
      console.warn('❌ GSR数据集[8]不存在');
      return;
    }
    
    const gsrPoints = [];
    for (const record of gsrData) {
      if (Number.isFinite(record.gsr)) {
        gsrPoints.push({ x: record.time, y: record.gsr });
      }
    }
    
    console.log(`🔄 VAChart.updateGSRData: 更新${gsrPoints.length}个GSR数据点`);
    console.log(`🔍 GSR数据样本: [${gsrPoints.slice(0, 3).map(p => `(${p.x.toFixed(1)}, ${p.y.toFixed(3)})`).join(', ')}]`);
    
    // 计算GSR数据的范围
    if (gsrPoints.length > 0) {
      const gsrValues = gsrPoints.map(p => p.y);
      const gsrMin = Math.min(...gsrValues);
      const gsrMax = Math.max(...gsrValues);
      console.log(`🔍 GSR数据范围: ${gsrMin.toFixed(3)} 到 ${gsrMax.toFixed(3)}`);
      
      // 获取右Y轴配置 - 适配旧版Chart.js格式
      let rightYAxis = null;
      
      // 尝试新版格式
      if (this._chart.options.scales && this._chart.options.scales['y1']) {
        rightYAxis = this._chart.options.scales['y1'];
        console.log(`🔍 找到新版右Y轴 (y1)`);
      }
      // 尝试旧版格式
      else if (this._chart.options.scales && this._chart.options.scales.yAxes && this._chart.options.scales.yAxes.length > 1) {
        rightYAxis = this._chart.options.scales.yAxes[1]; // 第二个Y轴
        console.log(`🔍 找到旧版右Y轴 (yAxes[1])`);
      }
      // 如果只有一个Y轴，也尝试使用它
      else if (this._chart.options.scales && this._chart.options.scales.yAxes && this._chart.options.scales.yAxes.length > 0) {
        rightYAxis = this._chart.options.scales.yAxes[0]; // 使用唯一的Y轴
        console.log(`🔍 使用主Y轴 (yAxes[0]) 显示GSR数据`);
      }
      
      console.log(`🔍 右Y轴对象:`, rightYAxis);
      
      if (rightYAxis) {
        // 保存原始范围用于调试
        const oldMin = rightYAxis.ticks ? rightYAxis.ticks.min : rightYAxis.min;
        const oldMax = rightYAxis.ticks ? rightYAxis.ticks.max : rightYAxis.max;
        
        // 检测是否为标准化数据 (均值接近0且范围较大)
        const gsrMean = gsrValues.reduce((a, b) => a + b, 0) / gsrValues.length;
        const isStandardizedData = Math.abs(gsrMean) < 0.1 && (gsrMax - gsrMin) > 5;
        console.log(`🔍 GSR数据分析: 均值=${gsrMean.toFixed(3)}, 范围=${(gsrMax - gsrMin).toFixed(3)}, 标准化=${isStandardizedData}`);
        
        let newMin, newMax;
        if (isStandardizedData) {
          // 标准化数据：使用数据实际范围加边距
          const padding = (gsrMax - gsrMin) * 0.1; // 10% 边距
          newMin = gsrMin - padding;
          newMax = gsrMax + padding;
          console.log(`🔄 右Y轴使用标准化动态范围: [${oldMin}, ${oldMax}] → [${newMin.toFixed(3)}, ${newMax.toFixed(3)}]`);
        } else {
          // 原始数据：使用动态范围
          const padding = (gsrMax - gsrMin) * 0.1; // 10% 边距
          newMin = gsrMin - padding;
          newMax = gsrMax + padding;
          console.log(`🔄 右Y轴使用原始动态范围: [${oldMin}, ${oldMax}] → [${newMin.toFixed(3)}, ${newMax.toFixed(3)}]`);
        }
        
        // 适配不同的Chart.js版本
        if (rightYAxis.ticks) {
          rightYAxis.ticks.min = newMin;
          rightYAxis.ticks.max = newMax;
        } else {
          rightYAxis.min = newMin;
          rightYAxis.max = newMax;
        }
      } else {
        console.warn('❌ 找不到任何可用的Y轴');
        console.log('🔍 完整的scales结构:', this._chart.options.scales);
      }
    }
    
    this._datasets[8].data = gsrPoints;
    
    // 强制更新图表，包括坐标轴
    this._chart.update('active');
    
    // 强制重绘图表
    if (this._chart.render) {
      this._chart.render();
    }
    
    // 额外尝试：强制重新计算Y轴范围
    if (gsrPoints.length > 0) {
      const gsrValues = gsrPoints.map(p => p.y);
      const gsrMin = Math.min(...gsrValues);
      const gsrMax = Math.max(...gsrValues);
      
      // 尝试通过Chart.js实例直接设置范围
      if (this._chart.scales) {
        const scaleIds = Object.keys(this._chart.scales);
        console.log(`🔍 运行时可用的scale IDs:`, scaleIds);
        
        // 专门更新GSR对应的Y轴
        const gsrScale = this._chart.scales['y-axis-gsr'];
        if (gsrScale) {
          // 检测是否为标准化数据
          const gsrMean = gsrValues.reduce((a, b) => a + b, 0) / gsrValues.length;
          const isStandardizedData = Math.abs(gsrMean) < 0.1 && (gsrMax - gsrMin) > 5;
          
          if (isStandardizedData) {
            // 标准化数据：使用数据实际范围加边距
            const padding = (gsrMax - gsrMin) * 0.1;
            gsrScale.options.min = gsrMin - padding;
            gsrScale.options.max = gsrMax + padding;
            console.log(`🔄 直接更新GSR专用scale (y-axis-gsr): 标准化动态范围 [${(gsrMin - padding).toFixed(3)}, ${(gsrMax + padding).toFixed(3)}]`);
          } else {
            // 原始数据：使用动态范围
            const padding = (gsrMax - gsrMin) * 0.1;
            gsrScale.options.min = gsrMin - padding;
            gsrScale.options.max = gsrMax + padding;
            console.log(`🔄 直接更新GSR专用scale (y-axis-gsr): 原始动态范围 [${(gsrMin - padding).toFixed(3)}, ${(gsrMax + padding).toFixed(3)}]`);
          }
        } else {
          // 备用：更新任何可用的Y轴
          for (const scaleId of scaleIds) {
            const scale = this._chart.scales[scaleId];
            if (scale.type === 'linear' && scale.axis === 'y') {
              // 检测是否为标准化数据
              const gsrMean = gsrValues.reduce((a, b) => a + b, 0) / gsrValues.length;
              const isStandardizedData = Math.abs(gsrMean) < 0.1 && (gsrMax - gsrMin) > 5;
              
              if (isStandardizedData) {
                // 标准化数据：使用数据实际范围加边距
                const padding = (gsrMax - gsrMin) * 0.1;
                scale.options.min = gsrMin - padding;
                scale.options.max = gsrMax + padding;
                console.log(`🔄 备用：直接更新scale ${scaleId}: 标准化动态范围 [${(gsrMin - padding).toFixed(3)}, ${(gsrMax + padding).toFixed(3)}]`);
              } else {
                // 原始数据：使用动态范围
                const padding = (gsrMax - gsrMin) * 0.1;
                scale.options.min = gsrMin - padding;
                scale.options.max = gsrMax + padding;
                console.log(`🔄 备用：直接更新scale ${scaleId}: 原始动态范围 [${(gsrMin - padding).toFixed(3)}, ${(gsrMax + padding).toFixed(3)}]`);
              }
              break;
            }
          }
        }
      }
    }
    
    // 额外尝试：完全重新渲染
    setTimeout(() => {
      this._chart.update('resize');
      console.log(`🔄 延迟更新完成`);
    }, 100);
    
    console.log(`✅ GSR图表已更新，包括Y轴范围`);
  }

  // 更新主Y轴范围（用于Valence/Arousal标准化后）
  updateMainAxisRange(valenceData, arousalData) {
    console.log(`🔄 VAChart.updateMainAxisRange: 更新主Y轴范围`);
    
    if (!valenceData || valenceData.length === 0) {
      console.log(`❌ Valence数据为空，跳过主Y轴更新`);
      return;
    }
    
    // 提取数值
    const valenceValues = valenceData.filter(record => 
      record.face && Number.isFinite(record.valence)
    ).map(record => record.valence);
    
    const arousalValues = arousalData ? arousalData.filter(record => 
      record.face && Number.isFinite(record.arousal)
    ).map(record => record.arousal) : [];
    
    if (valenceValues.length === 0) {
      console.log(`❌ 有效的Valence数据为空，跳过主Y轴更新`);
      return;
    }
    
    // 计算数据范围（包含Valence和Arousal）
    const allValues = [...valenceValues, ...arousalValues];
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    const padding = (maxValue - minValue) * 0.1; // 10% 边距
    
    console.log(`🔍 主Y轴数据范围: ${minValue.toFixed(3)} 到 ${maxValue.toFixed(3)}`);
    
    // 检测是否为标准化数据
    const mean = allValues.reduce((a, b) => a + b, 0) / allValues.length;
    const isStandardizedData = Math.abs(mean) < 0.1 && (maxValue - minValue) > 3;
    console.log(`🔍 主Y轴数据分析: 均值=${mean.toFixed(3)}, 范围=${(maxValue - minValue).toFixed(3)}, 标准化=${isStandardizedData}`);
    
    // 更新旧版Chart.js的主Y轴
    if (this._chart.options && this._chart.options.scales && this._chart.options.scales.yAxes) {
      const mainYAxis = this._chart.options.scales.yAxes[0]; // 主Y轴
      if (mainYAxis) {
        console.log(`🔍 找到旧版主Y轴 (yAxes[0])`);
        
        // 保存旧范围用于调试
        const oldMin = mainYAxis.ticks ? mainYAxis.ticks.min : 'undefined';
        const oldMax = mainYAxis.ticks ? mainYAxis.ticks.max : 'undefined';
        
        // 更新Y轴范围
        if (!mainYAxis.ticks) {
          mainYAxis.ticks = {};
        }
        
        if (isStandardizedData) {
          // 标准化数据：使用数据实际范围加边距
          mainYAxis.ticks.min = minValue - padding;
          mainYAxis.ticks.max = maxValue + padding;
          console.log(`🔄 主Y轴使用标准化动态范围: [${oldMin}, ${oldMax}] → [${(minValue - padding).toFixed(3)}, ${(maxValue + padding).toFixed(3)}]`);
        } else {
          // 原始数据：使用-1到1的固定范围
          mainYAxis.ticks.min = -1;
          mainYAxis.ticks.max = 1;
          console.log(`🔄 主Y轴使用原始固定范围: [${oldMin}, ${oldMax}] → [-1, 1]`);
        }
      }
    }
    
    // 更新新版Chart.js的主Y轴（如果存在）
    if (this._chart.scales) {
      const mainScale = this._chart.scales['y-axis-main'];
      if (mainScale) {
        console.log(`🔍 找到新版主Y轴 (y-axis-main)`);
        
        if (isStandardizedData) {
          // 标准化数据：使用数据实际范围加边距
          mainScale.options.min = minValue - padding;
          mainScale.options.max = maxValue + padding;
          console.log(`🔄 直接更新主Y轴: 标准化动态范围 [${(minValue - padding).toFixed(3)}, ${(maxValue + padding).toFixed(3)}]`);
        } else {
          // 原始数据：使用-1到1的固定范围
          mainScale.options.min = -1;
          mainScale.options.max = 1;
          console.log(`🔄 直接更新主Y轴: 原始固定范围 [-1, 1]`);
        }
      }
    }
    
    // 强制更新图表
    this._chart.update('active');
    
    // 强制重绘图表
    if (this._chart.render) {
      this._chart.render();
    }
    
    console.log(`✅ 主Y轴范围已更新`);
  }

  // 更新PPG数据
  updatePPGData(ppgData) {
    if (!this._datasets[9]) return;
    
    const ppgPoints = [];
    for (const record of ppgData) {
      if (Number.isFinite(record.ppg)) {
        ppgPoints.push({ x: record.time, y: record.ppg });
      }
    }
    
    this._datasets[9].data = ppgPoints;
    this._chart.update();
  }

  // 切换GSR显示
  toggleGSRDisplay(show) {
    if (this._datasets[8]) {
      const meta = this._chart.getDatasetMeta(8);
      meta.hidden = !show;
      this._config.options.scales.yAxes[1].display = show; // 显示GSR Y轴
      this._chart.update();
    }
  }

  // 切换PPG显示
  togglePPGDisplay(show) {
    if (this._datasets[9]) {
      const meta = this._chart.getDatasetMeta(9);
      meta.hidden = !show;
      this._config.options.scales.yAxes[2].display = show; // 显示PPG Y轴
      this._chart.update();
    }
  }

  // 清除生理信号数据
  clearPhysiologicalData() {
    if (this._datasets[8]) this._datasets[8].data = [];
    if (this._datasets[9]) this._datasets[9].data = [];
    this._chart.update();
  }

  // 更新异常标记显示
  updateAnomalyMarks(anomalyMarks) {
    console.log('🎯 VAChart.updateAnomalyMarks 被调用:', anomalyMarks);
    
    if (!this._datasets[10]) {
      console.warn('❌ 异常标记数据集不存在');
      return;
    }
    
    // 清空现有异常标记
    this._datasets[10].data = [];
    
    // 添加新的异常标记
    anomalyMarks.forEach(mark => {
      this._datasets[10].data.push({
        x: mark.time,
        y: mark.valence
      });
    });
    
    console.log(`⭐ 已更新 ${anomalyMarks.length} 个异常标记到图表`);
    this._chart.update();
  }

  // 清除异常标记
  clearAnomalyMarks() {
    if (this._datasets[10]) {
      this._datasets[10].data = [];
      this._chart.update();
    }
  }

  // 显示机器学习异常检测结果
  showMLAnomalies(anomalies, options = {}) {
    console.log('🤖 VAChart.showMLAnomalies 被调用:', anomalies.length, '个异常点', options);
    
    const { 
      color = '#ff6b6b', 
      size = 8, 
      label = 'ML异常点',
      dataSource = 'unknown'
    } = options;

    // 移除之前的ML异常点数据集
    this._datasets = this._datasets.filter(ds => !ds.label.includes('ML异常点'));
    
    // 调试信息
    console.log('🔍 异常点数据样本:', anomalies.slice(0, 3));
    console.log('🔍 Y值范围:', {
      min: Math.min(...anomalies.map(a => a.y)),
      max: Math.max(...anomalies.map(a => a.y))
    });
    console.log('🔍 数据源和Y轴ID判断:', {
      dataSource,
      dataSourceUpper: dataSource.toUpperCase(),
      isGSR: dataSource.toUpperCase() === 'GSR',
      targetYAxisID: dataSource.toUpperCase() === 'GSR' ? 'y-axis-gsr' : 'y-axis-main'
    });
    
    // 创建新的ML异常点数据集
    const mlAnomalyDataset = {
      label: `${label} (${dataSource})`,
      data: anomalies.map(anomaly => ({
        x: anomaly.x,
        y: anomaly.y
      })),
      backgroundColor: color,
      borderColor: color,
      pointRadius: size,
      pointHoverRadius: size + 2,
      pointBorderWidth: 2,
      pointBorderColor: '#fff',
      showLine: false,
      fill: false,
      lineTension: 0,
      borderWidth: 0,
      yAxisID: dataSource.toUpperCase() === 'GSR' ? 'y-axis-gsr' : 'y-axis-main' // 使用正确的Y轴ID
    };
    
    // 添加到数据集
    this._datasets.push(mlAnomalyDataset);
    this._chart.data.datasets = this._datasets;
    
    console.log(`⭐ 已添加 ${anomalies.length} 个ML异常点到图表，数据集索引: ${this._datasets.length - 1}`);
    console.log(`⭐ 当前总数据集数量: ${this._datasets.length}`);
    
    // 强制重新渲染
    this._chart.update('active');
  }

  // 隐藏机器学习异常检测结果
  hideMLAnomalies() {
    console.log('🤖 VAChart.hideMLAnomalies 被调用');
    
    // 移除ML异常点数据集
    this._datasets = this._datasets.filter(ds => !ds.label.includes('ML异常点'));
    this._chart.data.datasets = this._datasets;
    
    console.log('⭐ 已移除所有ML异常点');
    this._chart.update('none');
  }

  // 显示异常区间（作为半透明背景区域）
  showAnomalyIntervals(intervals, options = {}) {
    console.log('🔍 VAChart.showAnomalyIntervals 被调用:', intervals.length, '个异常区间');
    
    if (!intervals || intervals.length === 0) {
      console.log('没有异常区间数据，跳过显示');
      return;
    }

    // 统一样式配置 - 使用紫色避免与橙色GSR冲突
    const config = {
      color: 'rgba(139, 92, 246, 0.3)',    // 紫色背景
      borderColor: '#8b5cf6',              // 紫色边框
      borderWidth: 2,
      ...options
    };

    // 先移除之前的异常区间
    this.hideAnomalyIntervals();

    // 合并所有区间数据到一个数据集中，这样图例只显示一个条目
    const allIntervalData = [];
    
    intervals.forEach((interval, index) => {
      // 统一使用 startTime 和 endTime 属性
      const startTime = interval.startTime;
      const endTime = interval.endTime;
      
      // 为每个区间添加矩形数据点
      allIntervalData.push(
        { x: startTime, y: -1 },
        { x: startTime, y: 1 },
        { x: endTime, y: 1 },
        { x: endTime, y: -1 },
        { x: startTime, y: null } // 分隔不同区间
      );
      console.log(`🔶 添加异常区间${index + 1}: ${startTime.toFixed(1)}s-${endTime.toFixed(1)}s`);
    });

    // 创建单一数据集包含所有区间
    const dataset = {
      label: '🔶 异常区间',
      data: allIntervalData,
      fill: true,
      backgroundColor: config.color,
      borderColor: config.borderColor,
      borderWidth: config.borderWidth,
      pointRadius: 0,
      showLine: true,
      lineTension: 0
    };

    this._datasets.push(dataset);

    // 更新图表
    this._chart.data.datasets = this._datasets;
    this._chart.update('active');

    console.log(`✅ 显示了 ${intervals.length} 个异常区间`);
  }

  // 隐藏异常区间
  hideAnomalyIntervals() {
    // 移除异常区间数据集
    this._datasets = this._datasets.filter(ds => !ds.label.includes('异常区间'));
    this._chart.data.datasets = this._datasets;
    
    console.log('🗑️ 已移除所有异常区间');
    this._chart.update('none');
  }

}

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

export { VAChart };