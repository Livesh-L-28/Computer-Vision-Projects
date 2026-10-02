// Frontend Logic for Computer Vision Intelligence Suite Dashboard

document.addEventListener('DOMContentLoaded', () => {
  let currentSystem = 'combat';
  let pollInterval = null;
  let isStreaming = false;

  // DOM Elements
  const tabs = document.querySelectorAll('.sys-tab');
  const activeSystemTag = document.getElementById('active-system-tag');
  const form = document.getElementById('pipeline-form');
  const radioSynthetic = document.getElementById('mode-synthetic');
  const radioUpload = document.getElementById('mode-upload');
  const uploadGroup = document.getElementById('upload-group');
  const frameSlider = document.getElementById('frame-slider');
  const frameVal = document.getElementById('frame-val');
  const btnRun = document.getElementById('btn-run');
  const btnStream = document.getElementById('btn-stream');
  const statusIndicator = document.getElementById('status-indicator');
  const statusText = document.getElementById('status-text');

  // Telemetry elements
  const teleStatus = document.getElementById('tele-status');
  const teleTargets = document.getElementById('tele-targets');
  
  // Viewport elements
  const videoPlayer = document.getElementById('output-video-player');
  const videoSource = document.getElementById('video-source');
  const streamPlayer = document.getElementById('stream-player');
  const videoOverlayMsg = document.getElementById('video-overlay-msg');
  const viewportState = document.getElementById('viewport-state');

  // Artifact elements
  const artifactsContainer = document.getElementById('artifacts-container');
  const heatmapCard = document.getElementById('heatmap-card');
  const heatmapPreview = document.getElementById('heatmap-preview');
  const dashboardCard = document.getElementById('dashboard-card');
  const dashboardPreview = document.getElementById('dashboard-preview');
  const btnDownloadVideo = document.getElementById('btn-download-video');
  const btnDownloadCsv = document.getElementById('btn-download-csv');

  // Tab switching
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentSystem = tab.dataset.system;
      
      const systemTitles = {
        combat: 'AIR COMBAT INTELLIGENCE',
        runway: 'AIRPORT RUNWAY SURFACE',
        anpr: 'LICENSE PLATE ANPR'
      };
      activeSystemTag.textContent = systemTitles[currentSystem] || currentSystem.toUpperCase();

      // If active streaming, switch stream source
      if (isStreaming) {
        streamPlayer.src = `/api/stream/${currentSystem}?t=${Date.now()}`;
      }
    });
  });

  // Radio toggle for input source
  radioSynthetic.addEventListener('change', () => {
    uploadGroup.style.display = 'none';
  });

  radioUpload.addEventListener('change', () => {
    uploadGroup.style.display = 'flex';
  });

  // Slider value feedback
  frameSlider.addEventListener('input', (e) => {
    frameVal.textContent = e.target.value;
  });

  // Live Stream Button Handler
  btnStream.addEventListener('click', () => {
    if (!isStreaming) {
      isStreaming = true;
      btnStream.classList.remove('btn-outline');
      btnStream.classList.add('btn-primary');
      btnStream.textContent = '⏹ STOP LIVE FEED';

      videoPlayer.pause();
      videoPlayer.style.display = 'none';
      streamPlayer.style.display = 'block';
      streamPlayer.src = `/api/stream/${currentSystem}?t=${Date.now()}`;
      videoOverlayMsg.style.display = 'none';

      statusText.textContent = 'LIVE FEED STREAMING';
      statusIndicator.style.backgroundColor = '#00f2fe';
      statusIndicator.style.boxShadow = '0 0 12px #00f2fe';
      teleStatus.textContent = 'STREAMING';
      viewportState.textContent = 'REAL-TIME MJPEG';
    } else {
      isStreaming = false;
      btnStream.classList.remove('btn-primary');
      btnStream.classList.add('btn-outline');
      btnStream.textContent = '📡 LIVE VIDEO FEED';

      streamPlayer.src = '';
      streamPlayer.style.display = 'none';
      videoPlayer.style.display = 'block';
      videoOverlayMsg.style.display = 'flex';

      statusText.textContent = 'SYSTEM READY';
      statusIndicator.style.backgroundColor = '#39ff14';
      statusIndicator.style.boxShadow = '0 0 8px #39ff14';
      teleStatus.textContent = 'IDLE';
      viewportState.textContent = 'STANDBY';
    }
  });

  // Form submission handler
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isStreaming) {
      btnStream.click(); // Stop stream if running
    }

    const isSynthetic = radioSynthetic.checked;
    const frames = frameSlider.value;

    btnRun.disabled = true;
    btnRun.innerHTML = '⏳ INITIATING MODEL...';
    statusText.textContent = 'EXECUTING PIPELINE';
    statusIndicator.style.backgroundColor = '#fcd34d';
    statusIndicator.style.boxShadow = '0 0 10px #fcd34d';
    teleStatus.textContent = 'RUNNING';
    viewportState.textContent = 'PROCESSING';
    
    streamPlayer.style.display = 'none';
    videoPlayer.style.display = 'block';
    videoOverlayMsg.style.display = 'flex';
    videoOverlayMsg.querySelector('p').textContent = 'Inference engine active...';
    artifactsContainer.style.display = 'none';

    try {
      if (isSynthetic) {
        const formData = new FormData();
        formData.append('system', currentSystem);
        formData.append('frames', frames);

        const res = await fetch('/api/run-synthetic', {
          method: 'POST',
          body: formData
        });
        if (!res.ok) throw new Error('Simulation start request failed.');
      } else {
        const fileInput = document.getElementById('video-file');
        if (!fileInput.files.length) {
          alert('Please select a video file to upload.');
          resetRunButton();
          return;
        }
        const formData = new FormData();
        formData.append('system', currentSystem);
        formData.append('frames', frames);
        formData.append('file', fileInput.files[0]);

        const res = await fetch('/api/run-upload', {
          method: 'POST',
          body: formData
        });
        if (!res.ok) throw new Error('Upload processing request failed.');
      }

      startStatusPolling();

    } catch (err) {
      alert(`Pipeline error: ${err.message}`);
      resetRunButton();
    }
  });

  function startStatusPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(checkStatus, 1500);
  }

  async function checkStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();

      teleStatus.textContent = data.status.toUpperCase();

      if (data.metrics && Object.keys(data.metrics).length > 0) {
        const firstKey = Object.keys(data.metrics)[0];
        teleTargets.textContent = `${data.metrics[firstKey]}`;
      }

      if (data.status === 'completed') {
        clearInterval(pollInterval);
        resetRunButton();
        statusText.textContent = 'ANALYSIS COMPLETE';
        statusIndicator.style.backgroundColor = '#39ff14';
        statusIndicator.style.boxShadow = '0 0 10px #39ff14';
        viewportState.textContent = 'ANNOTATED PLAYBACK';

        // Load Output Video
        if (data.latest_output_video) {
          videoOverlayMsg.style.display = 'none';
          videoSource.src = `${data.latest_output_video}?t=${Date.now()}`;
          videoPlayer.load();
          videoPlayer.play().catch(() => {});
          btnDownloadVideo.href = data.latest_output_video;
        }

        // Show artifacts
        artifactsContainer.style.display = 'grid';

        if (data.latest_heatmap) {
          heatmapCard.style.display = 'flex';
          heatmapPreview.src = `${data.latest_heatmap}?t=${Date.now()}`;
        } else {
          heatmapCard.style.display = 'none';
        }

        if (data.latest_dashboard) {
          dashboardCard.style.display = 'flex';
          dashboardPreview.src = `${data.latest_dashboard}?t=${Date.now()}`;
        } else {
          dashboardCard.style.display = 'none';
        }

        if (data.latest_report_csv) {
          btnDownloadCsv.href = data.latest_report_csv;
          btnDownloadCsv.style.display = 'inline-flex';
        } else {
          btnDownloadCsv.style.display = 'none';
        }

      } else if (data.status === 'error') {
        clearInterval(pollInterval);
        resetRunButton();
        statusText.textContent = 'EXECUTION ERROR';
        statusIndicator.style.backgroundColor = '#ff3366';
        statusIndicator.style.boxShadow = '0 0 10px #ff3366';
        alert(`Error during processing: ${data.metrics?.Error || 'Unknown error'}`);
      }

    } catch (e) {
      console.warn('Status poll exception:', e);
    }
  }

  function resetRunButton() {
    btnRun.disabled = false;
    btnRun.innerHTML = '<span class="btn-glow"></span>▶ EXECUTE PIPELINE';
  }
});
