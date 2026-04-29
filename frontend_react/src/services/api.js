/**
 * API Service for GRACE Downscaling Engine
 * Handles all communication with FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/**
 * Base fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, defaultOptions);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || `HTTP error! status: ${response.status}`);
    }

    // Check if backend returned an error status in the response body
    if (data.status === 'error') {
      throw new Error(data.message || 'Backend error occurred');
    }

    return data;
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    throw error;
  }
}

/**
 * Poll a status endpoint until task completes
 */
async function pollTaskStatus(taskId, sessionId, statusEndpoint, resultEndpoint, onProgress) {
  const pollInterval = 2000; // 2 seconds
  const maxAttempts = 300; // 10 minutes max
  let attempts = 0;

  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        attempts++;
        
        if (attempts > maxAttempts) {
          reject(new Error('Task timeout - exceeded maximum wait time'));
          return;
        }

        const statusData = await apiFetch(
          `${statusEndpoint}/${taskId}?session_id=${sessionId}`
        );

        const { status, progress, error } = statusData.data;

        // Call progress callback
        if (onProgress && progress !== undefined) {
          onProgress(progress, status);
        }

        if (status === 'completed' || status === 'complete') {
          // Get final results
          const resultData = await apiFetch(
            `${resultEndpoint}/${taskId}?session_id=${sessionId}`
          );
          resolve(resultData.data);
        } else if (status === 'failed' || status === 'error') {
          reject(new Error(error || 'Task failed'));
        } else {
          // Continue polling
          setTimeout(poll, pollInterval);
        }
      } catch (error) {
        reject(error);
      }
    };

    poll();
  });
}

// ============================================================================
// SESSION MANAGEMENT
// ============================================================================

export const sessionApi = {
  /**
   * Create a new session
   */
  async create() {
    return await apiFetch('/api/session/create', {
      method: 'POST',
    });
  },

  /**
   * Get session info
   */
  async get(sessionId) {
    return await apiFetch(`/api/session/${sessionId}`);
  },

  /**
   * Delete session
   */
  async delete(sessionId) {
    return await apiFetch(`/api/session/${sessionId}`, {
      method: 'DELETE',
    });
  },
};

// ============================================================================
// TAB 1: GEOGRAPHIC SETUP
// ============================================================================

export const setupApi = {
  /**
   * Validate geographic bounds
   */
  async validateBounds(sessionId, bounds) {
    return await apiFetch('/api/setup/validate-bounds', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        ...bounds,
      }),
    });
  },

  /**
   * Load GRACE and ERA5 data
   */
  async loadData(sessionId, bounds, graceDataset = 'CSR') {
    return await apiFetch('/api/setup/load-data', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        grace_dataset: graceDataset,
        ...bounds,
      }),
    });
  },

  /**
   * Get coastlines GeoJSON
   */
  async getCoastlines() {
    return await apiFetch('/api/setup/coastlines');
  },
};

// ============================================================================
// TAB 2: DATA PROCESSING & RFE
// ============================================================================

export const dataProcessingApi = {
  /**
   * Prepare data for RFE
   */
  async prepareData(sessionId) {
    return await apiFetch('/api/data-processing/prep', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
      }),
    });
  },

  /**
   * Run RFE (Recursive Feature Elimination)
   * @param {Function} onProgress - Callback function (progress, status) => void
   */
  async runRFE(sessionId, modelType, nFeatures, onProgress) {
    // Start RFE task
    const startData = await apiFetch('/api/data-processing/rfe', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        model_type: modelType,
        n_features: nFeatures,
      }),
    });

    const taskId = startData.data.task_id;

    // Poll for completion
    return await pollTaskStatus(
      taskId,
      sessionId,
      '/api/data-processing/status',
      '/api/data-processing/result',
      onProgress
    );
  },
};

// ============================================================================
// TAB 3: MODEL TRAINING
// ============================================================================

export const trainingApi = {
  /**
   * Start model training
   * @param {Function} onProgress - Callback function (progress, status) => void
   */
  async trainModel(sessionId, model, trainType, onProgress) {
    // Start training task
    const startData = await apiFetch('/api/training/start', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        model: model,
        train_type: trainType,
      }),
    });

    const taskId = startData.data.task_id;

    // Poll for completion
    return await pollTaskStatus(
      taskId,
      sessionId,
      '/api/training/status',
      '/api/training/result',
      onProgress
    );
  },

  /**
   * Upload pre-trained model
   */
  async uploadModel(sessionId, file) {
    const formData = new FormData();
    formData.append('file', file);

    return await apiFetch(`/api/training/upload-model?session_id=${sessionId}`, {
      method: 'POST',
      headers: {}, // Let browser set Content-Type for FormData
      body: formData,
    });
  },

  /**
   * Get URL to download trained model
   */
  getDownloadModelUrl(sessionId) {
    return `${API_BASE_URL}/api/training/download-model?session_id=${sessionId}`;
  },
};

// ============================================================================
// TAB 4: MAPS & VISUALIZATION
// ============================================================================

export const mapsApi = {
  /**
   * Generate ERA5 variable map
   * @param {Function} onProgress - Callback function (progress, status) => void
   */
  async generateERA5Map(sessionId, variable, year, month, onProgress) {
    // Start map generation
    const startData = await apiFetch('/api/maps/era5', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        variable: variable,
        year: year,
        month: month,
      }),
    });

    const taskId = startData.data.task_id;

    // Poll for completion
    const result = await pollTaskStatus(
      taskId,
      sessionId,
      '/api/maps/status',
      '/api/maps/status', // Status endpoint returns map_id
      onProgress
    );

    // Download the map
    const mapData = await apiFetch(
      `/api/maps/download/${result.result.map_id}?session_id=${sessionId}`
    );

    return mapData.data;
  },

  /**
   * Generate GRACE comparison map
   * @param {Function} onProgress - Callback function (progress, status) => void
   */
  async generateGRACEMap(sessionId, year, month, onProgress) {
    // Start map generation
    const startData = await apiFetch('/api/maps/grace-comparison', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        year: year,
        month: month,
      }),
    });

    const taskId = startData.data.task_id;

    // Poll for completion
    const result = await pollTaskStatus(
      taskId,
      sessionId,
      '/api/maps/status',
      '/api/maps/status',
      onProgress
    );

    // Download the map
    const mapData = await apiFetch(
      `/api/maps/download/${result.result.map_id}?session_id=${sessionId}`
    );

    return mapData.data;
  },
};

// ============================================================================
// TAB 5: STATISTICAL ANALYSIS
// ============================================================================

export const analysisApi = {
  /**
   * Evaluate model at specific location
   */
  async evaluateModel(sessionId, latitude, longitude, startYear, endYear) {
    return await apiFetch('/api/analysis/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        latitude: latitude,
        longitude: longitude,
        start_year: startYear,
        end_year: endYear,
      }),
    });
  },

  /**
   * Get feature importance
   */
  async getFeatureImportance(sessionId) {
    return await apiFetch('/api/analysis/feature-importance', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
      }),
    });
  },
};

// ============================================================================
// UTILITY ENDPOINTS
// ============================================================================

export const utilityApi = {
  /**
   * Check server health
   */
  async health() {
    return await apiFetch('/api/health');
  },

  /**
   * Get system configuration
   */
  async config() {
    return await apiFetch('/api/config');
  },
};

// Default export with all APIs
export default {
  session: sessionApi,
  setup: setupApi,
  dataProcessing: dataProcessingApi,
  training: trainingApi,
  maps: mapsApi,
  analysis: analysisApi,
  utility: utilityApi,
};
