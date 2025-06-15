import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Divider,
  IconButton,
  useTheme,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { styled } from '@mui/material/styles';
import { Message } from '../types/chat';

const ChatContainer = styled(Paper)({
  display: 'flex',
  flexDirection: 'column',
  height: 'calc(100vh - 200px)',
  maxWidth: '1200px',
  margin: '0 auto',
  overflow: 'hidden',
});

const MessagesContainer = styled(Box)({
  flex: 1,
  overflowY: 'auto',
  padding: '16px',
});

const InputContainer = styled(Box)({
  display: 'flex',
  padding: '16px',
  borderTop: '1px solid #e0e0e0',
  backgroundColor: '#f5f5f5',
});

const MessageBubble = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'isUser',
})<{ isUser: boolean }>(({ theme, isUser }) => ({
  maxWidth: '70%',
  padding: '8px 16px',
  borderRadius: '18px',
  marginBottom: '8px',
  alignSelf: isUser ? 'flex-end' : 'flex-start',
  backgroundColor: isUser ? theme.palette.primary.main : '#e9ecef',
  color: isUser ? '#ffffff' : '#212529',
}));

const HomePage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();

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
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h4" gutterBottom>
        Genomic Guidelines Assistant
      </Typography>
      
      <ChatContainer elevation={3}>
        <MessagesContainer>
          <List>
            {messages.map((message) => (
              <React.Fragment key={message.id}>
                <ListItem sx={{ px: 0, py: 0.5 }}>
                  <MessageBubble isUser={message.role === 'user'}>
                    <ListItemText
                      primary={message.text}
                      primaryTypographyProps={{
                        color: message.role === 'user' ? 'inherit' : 'text.primary',
                      }}
                    />
                    {message.sources && message.sources.length > 0 && (
                      <Box sx={{ mt: 1, fontSize: '0.75rem' }}>
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="caption" color="text.secondary">
                          Sources:
                        </Typography>
                        {message.sources.map((source, idx) => (
                          <Typography key={idx} variant="caption" display="block">
                            • {source.title}
                          </Typography>
                        ))}
                      </Box>
                    )}
                  </MessageBubble>
                </ListItem>
              </React.Fragment>
            ))}
            {isLoading && (
              <ListItem sx={{ justifyContent: 'center' }}>
                <CircularProgress size={24} />
              </ListItem>
            )}
            <div ref={messagesEndRef} />
          </List>
        </MessagesContainer>
        
        <InputContainer>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Ask me anything about genomic guidelines..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            multiline
            maxRows={4}
            sx={{ mr: 1 }}
          />
          <IconButton
            color="primary"
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
            sx={{ alignSelf: 'flex-end' }}
          >
            <SendIcon />
          </IconButton>
        </InputContainer>
      </ChatContainer>
    </Box>
  );
};

export default HomePage;
