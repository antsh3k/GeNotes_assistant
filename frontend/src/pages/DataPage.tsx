import React, { useState, useRef, ChangeEvent } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Divider,
  CircularProgress,
  Alert,
  Snackbar,
  useTheme,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Link as LinkIcon,
  InsertDriveFile as FileIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { documentApi } from '../services/api';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
}));

const UploadArea = styled(Box)(({ theme, isDragging }: { isDragging: boolean; theme: any }) => ({
  border: `2px dashed ${isDragging ? theme.palette.primary.main : theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  backgroundColor: isDragging ? 'rgba(0, 94, 184, 0.05)' : 'transparent',
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: theme.palette.primary.main,
    backgroundColor: 'rgba(0, 94, 184, 0.05)',
  },
}));

const FileList = styled(List)({
  maxHeight: '300px',
  overflowY: 'auto',
  marginTop: '16px',
});

interface Document {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'uploading' | 'success' | 'error';
  error?: string;
}

const DataPage: React.FC = () => {
  const [url, setUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const theme = useTheme();

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const files = Array.from(event.target.files);
      handleFiles(files);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      handleFiles(files);
    }
  };

  const handleFiles = async (files: File[] | FileList) => {
    // Convert FileList to array if it's not already an array
    const filesArray: File[] = Array.isArray(files) ? files : Array.from(files);
    const newDocuments: Document[] = filesArray.map((file: File) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'uploading' as const,
    }));

    setDocuments((prev) => [...prev, ...newDocuments]);

    // Upload files one by one
    for (let i = 0; i < filesArray.length; i++) {
      const file = filesArray[i];
      try {
        // TODO: Uncomment when backend is ready
        // await documentApi.uploadDocument(file);
        
        // Simulate upload
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.id === newDocuments[i].id
              ? { ...doc, status: 'success' as const }
              : doc
          )
        );
        
        showSnackbar('File uploaded successfully', 'success');
      } catch (error) {
        console.error('Error uploading file:', error);
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.id === newDocuments[i].id
              ? {
                  ...doc,
                  status: 'error' as const,
                  error: 'Failed to upload file',
                }
              : doc
          )
        );
        showSnackbar('Failed to upload file', 'error');
      }
    }
  };

  const handleScrapeUrl = async () => {
    if (!url.trim()) return;
    
    setIsLoading(true);
    
    try {
      // TODO: Uncomment when backend is ready
      // await documentApi.scrapeUrl(url);
      
      // Simulate scraping
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      showSnackbar('URL scraped successfully', 'success');
      setUrl('');
    } catch (error) {
      console.error('Error scraping URL:', error);
      showSnackbar('Failed to scrape URL', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteDocument = (id: string) => {
    // TODO: Implement delete document API call
    setDocuments((prev) => prev.filter((doc) => doc.id !== id));
    showSnackbar('Document deleted', 'success');
  };

  const showSnackbar = (message: string, severity: 'success' | 'error') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar((prev) => ({ ...prev, open: false }));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Management
      </Typography>
      
      <StyledPaper>
        <Typography variant="h6" gutterBottom>
          Upload Documents
        </Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Upload PDF, DOCX, or TXT files to be processed and added to the knowledge base.
        </Typography>
        
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          multiple
          accept=".pdf,.docx,.txt"
          style={{ display: 'none' }}
        />
        
        <UploadArea
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          isDragging={isDragging}
          theme={theme}
        >
          <CloudUploadIcon fontSize="large" color="action" style={{ fontSize: 48 }} />
          <Typography variant="body1" gutterBottom>
            Drag & drop files here or click to browse
          </Typography>
          <Typography variant="caption" color="textSecondary">
            Supported formats: PDF, DOCX, TXT (max 20MB)
          </Typography>
        </UploadArea>
        
        {documents.length > 0 && (
          <>
            <Typography variant="subtitle2" style={{ margin: '16px 0 8px' }}>
              Upload Queue ({documents.length})
            </Typography>
            <FileList>
              {documents.map((doc) => (
                <React.Fragment key={doc.id}>
                  <ListItem
                    secondaryAction={
                      <IconButton
                        edge="end"
                        aria-label="delete"
                        onClick={() => handleDeleteDocument(doc.id)}
                        disabled={doc.status === 'uploading'}
                      >
                        <DeleteIcon />
                      </IconButton>
                    }
                  >
                    <ListItemIcon>
                      {doc.status === 'success' ? (
                        <CheckCircleIcon color="success" />
                      ) : doc.status === 'error' ? (
                        <ErrorIcon color="error" />
                      ) : (
                        <CircularProgress size={24} />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={doc.name}
                      secondary={`${formatFileSize(doc.size)} • ${doc.type}`}
                      primaryTypographyProps={{
                        color: doc.status === 'error' ? 'error' : 'textPrimary',
                      }}
                      secondaryTypographyProps={{
                        color: doc.status === 'error' ? 'error' : 'textSecondary',
                      }}
                    />
                  </ListItem>
                  <Divider component="li" />
                </React.Fragment>
              ))}
            </FileList>
          </>
        )}
      </StyledPaper>
      
      <StyledPaper>
        <Typography variant="h6" gutterBottom>
          Scrape Web Content
        </Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Enter a URL to scrape content and add it to the knowledge base.
        </Typography>
        
        <Box display="flex" gap={2} alignItems="center">
          <TextField
            fullWidth
            variant="outlined"
            placeholder="https://example.com"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleScrapeUrl()}
            InputProps={{
              startAdornment: <LinkIcon color="action" style={{ marginRight: 8 }} />,
            }}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleScrapeUrl}
            disabled={!url.trim() || isLoading}
            startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : null}
          >
            {isLoading ? 'Scraping...' : 'Scrape'}
          </Button>
        </Box>
      </StyledPaper>
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DataPage;
