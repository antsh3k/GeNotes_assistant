export interface MessageSource {
  title: string;
  url: string;
}

export interface Message {
  id: number;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string;
  sources?: MessageSource[];
}

export interface ChatRequest {
  message: string;
  history?: Message[];
  collectionName?: string;
}

export interface ChatResponse {
  response: string;
  sources?: MessageSource[];
  metadata?: {
    collection: string;
    model: string;
    timestamp: string;
  };
}
