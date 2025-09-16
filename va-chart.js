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
    if (!this._datasets[8]) return;
    
    const gsrPoints = [];
    for (const record of gsrData) {
      if (Number.isFinite(record.gsr)) {
        gsrPoints.push({ x: record.time, y: record.gsr });
      }
    }
    
    this._datasets[8].data = gsrPoints;
    this._chart.update();
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

}

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

export { VAChart };