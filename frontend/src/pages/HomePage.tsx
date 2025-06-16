import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  CircularProgress,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Message } from '../types/chat';

const MainContainer = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: '#f5f5f5',
  minHeight: 'calc(100vh - 100px)', // Adjust for app header
  padding: '20px',
});

const ContentContainer = styled(Box)({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  maxWidth: '1000px',
  margin: '0 auto',
  width: '100%',
});

const ChatContainer = styled(Paper)({
  display: 'flex',
  flexDirection: 'column',
  height: '600px',
  overflow: 'hidden',
  borderRadius: '8px',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
});

const MessagesContainer = styled(Box)({
  flex: 1,
  overflowY: 'auto',
  padding: '20px',
  backgroundColor: 'white',
});

const InputContainer = styled(Box)({
  display: 'flex',
  padding: '16px 20px',
  borderTop: '1px solid #e0e0e0',
  backgroundColor: 'white',
  gap: '12px',
  alignItems: 'flex-end',
});

const MessageBubble = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'isUser',
})<{ isUser: boolean }>(({ theme, isUser }) => ({
  maxWidth: '80%',
  padding: '12px 16px',
  borderRadius: isUser ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
  marginBottom: '12px',
  alignSelf: isUser ? 'flex-end' : 'flex-start',
  backgroundColor: isUser ? '#4db6e6' : '#e9ecef', // Light blue for user messages
  color: isUser ? '#ffffff' : '#333333',
  marginLeft: isUser ? 'auto' : '0',
  marginRight: isUser ? '0' : 'auto',
}));

const SendButton = styled(Button)({
  backgroundColor: '#4caf50', // Green send button
  color: 'white',
  minWidth: '80px',
  height: '40px',
  '&:hover': {
    backgroundColor: '#45a049',
  },
  '&:disabled': {
    backgroundColor: '#cccccc',
  },
});

const HomePage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello, I'm the GeNotes Assistant. I can help you find and understand genomic clinical guidelines. How can I assist you today?",
      role: 'assistant',
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      text: input,
      role: 'user',
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // TODO: Replace with actual API call
      // const response = await axios.post('/api/chat', { message: input });
      // const botMessage: Message = {
      //   id: Date.now() + 1,
      //   text: response.data.response,
      //   role: 'assistant',
      //   timestamp: new Date().toISOString(),
      //   sources: response.data.sources
      // };
      
      // Simulate API call
      setTimeout(() => {
        const botMessage: Message = {
          id: Date.now() + 1,
          text: "I'm your GeNotes assistant. How can I help you with genomic guidelines today?",
          role: 'assistant',
          timestamp: new Date().toISOString(),
          sources: [
            { title: 'NHS Genomic Medicine Service', url: '#' },
            { title: 'Genomics England', url: '#' },
          ],
        };
        setMessages((prev) => [...prev, botMessage]);
        setIsLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <MainContainer>
      <ContentContainer>
        <Typography 
          variant="h4" 
          gutterBottom 
          sx={{ 
            color: '#005eb8', 
            fontWeight: 'bold',
            textAlign: 'center',
            marginBottom: '30px'
          }}
        >
          GeNotes Genomic Guidelines Assistant
        </Typography>
        
        <ChatContainer elevation={3}>
          <MessagesContainer>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              {messages.map((message) => (
                <React.Fragment key={message.id}>
                  <MessageBubble isUser={message.role === 'user'}>
                    <Typography 
                      sx={{ 
                        whiteSpace: 'pre-line',
                        lineHeight: 1.5,
                        fontSize: '14px'
                      }}
                    >
                      {message.text}
                    </Typography>
                    {message.sources && message.sources.length > 0 && (
                      <Box sx={{ mt: 2, pt: 1, borderTop: '1px solid rgba(255,255,255,0.3)' }}>
                        <Typography variant="caption" sx={{ fontSize: '11px', opacity: 0.8 }}>
                          Source: {message.sources[0].title}
                        </Typography>
                      </Box>
                    )}
                  </MessageBubble>
                </React.Fragment>
              ))}
              {isLoading && (
                <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                  <CircularProgress size={24} />
                </Box>
              )}
              <div ref={messagesEndRef} />
            </Box>
          </MessagesContainer>
          
          <InputContainer>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Type your question about genomic guidelines..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
              multiline
              maxRows={4}
              size="small"
              sx={{ 
                '& .MuiOutlinedInput-root': {
                  borderRadius: '20px',
                  backgroundColor: '#f9f9f9',
                }
              }}
            />
            <SendButton
              onClick={handleSendMessage}
              disabled={!input.trim() || isLoading}
              variant="contained"
            >
              Send
            </SendButton>
          </InputContainer>
        </ChatContainer>
      </ContentContainer>
    </MainContainer>
  );
};

export default HomePage;
