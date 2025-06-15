import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { ChatRequest, ChatResponse, Message } from '../types/chat';

const api: AxiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

// Request interceptor for API calls
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for API calls
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API methods
export const chatApi = {
  sendMessage: async (message: string, history: Message[] = []): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/api/chat', {
      message,
      history,
    });
    return response.data;
  },
};

export const documentApi = {
  uploadDocument: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },
  
  scrapeUrl: async (url: string) => {
    const response = await api.post('/api/scrape', { url });
    return response.data;
  },
  
  listDocuments: async () => {
    const response = await api.get('/api/documents');
    return response.data;
  },
};

export const vectorStoreApi = {
  listCollections: async () => {
    const response = await api.get('/api/collections');
    return response.data;
  },
  
  createCollection: async (name: string) => {
    const response = await api.post('/api/collections', { name });
    return response.data;
  },
  
  deleteCollection: async (name: string) => {
    const response = await api.delete(`/api/collections/${name}`);
    return response.data;
  },
};

export const authApi = {
  login: async (username: string, password: string) => {
    const response = await api.post('/api/auth/login', { username, password });
    return response.data;
  },
  
  logout: async () => {
    localStorage.removeItem('authToken');
    // Optionally call backend to invalidate token
    // await api.post('/api/auth/logout');
  },
};

export default api;
