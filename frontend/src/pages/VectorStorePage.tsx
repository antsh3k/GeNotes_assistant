import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  ListItemIcon,
  Switch,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Divider,
  CircularProgress,
  Tooltip,
  useTheme,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Storage as StorageIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { vectorStoreApi } from '../services/api';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
}));

const CollectionList = styled(List)({
  maxHeight: '500px',
  overflowY: 'auto',
  marginTop: '16px',
});

interface Collection {
  name: string;
  count: number;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

const VectorStorePage: React.FC = () => {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [editingCollection, setEditingCollection] = useState<Collection | null>(null);
  const [collectionName, setCollectionName] = useState('');
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' }>({
    open: false,
    message: '',
    severity: 'info',
  });
  const theme = useTheme();

  useEffect(() => {
    fetchCollections();
  }, []);

  const fetchCollections = async () => {
    setIsLoading(true);
    try {
      // TODO: Uncomment when backend is ready
      // const data = await vectorStoreApi.listCollections();
      // setCollections(data.collections);
      
      // Mock data for now
      await new Promise(resolve => setTimeout(resolve, 800));
      const mockCollections: Collection[] = [
        {
          name: 'genomic_guidelines',
          count: 1245,
          isActive: true,
          createdAt: '2023-06-10T14:30:00Z',
          updatedAt: '2023-06-15T09:15:00Z',
        },
        {
          name: 'research_papers',
          count: 532,
          isActive: false,
          createdAt: '2023-05-28T11:20:00Z',
          updatedAt: '2023-06-12T16:45:00Z',
        },
      ];
      setCollections(mockCollections);
    } catch (error) {
      console.error('Error fetching collections:', error);
      showSnackbar('Failed to load collections', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleOpenDialog = (collection: Collection | null = null) => {
    if (collection) {
      setEditingCollection(collection);
      setCollectionName(collection.name);
    } else {
      setEditingCollection(null);
      setCollectionName('');
    }
    setIsDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setIsDialogOpen(false);
    setEditingCollection(null);
    setCollectionName('');
  };

  const handleSaveCollection = async () => {
    if (!collectionName.trim()) {
      showSnackbar('Collection name cannot be empty', 'error');
      return;
    }

    try {
      if (editingCollection) {
        // TODO: Implement update collection API call
        // await vectorStoreApi.updateCollection(editingCollection.name, { name: collectionName });
        
        // Mock update
        await new Promise(resolve => setTimeout(resolve, 800));
        
        setCollections(prev =>
          prev.map(coll =>
            coll.name === editingCollection.name
              ? { ...coll, name: collectionName, updatedAt: new Date().toISOString() }
              : coll
          )
        );
        
        showSnackbar('Collection updated successfully', 'success');
      } else {
        // TODO: Implement create collection API call
        // await vectorStoreApi.createCollection(collectionName);
        
        // Mock create
        await new Promise(resolve => setTimeout(resolve, 800));
        
        const newCollection: Collection = {
          name: collectionName,
          count: 0,
          isActive: false,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        
        setCollections(prev => [...prev, newCollection]);
        showSnackbar('Collection created successfully', 'success');
      }
      
      handleCloseDialog();
    } catch (error) {
      console.error('Error saving collection:', error);
      const errorMessage = error instanceof Error 
        ? error.message 
        : 'Failed to save collection';
      showSnackbar(errorMessage, 'error');
    }
  };

  const handleDeleteCollection = async (collectionName: string) => {
    if (!window.confirm(`Are you sure you want to delete the collection "${collectionName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      setIsDeleting(true);
      
      // TODO: Implement delete collection API call
      // await vectorStoreApi.deleteCollection(collectionName);
      
      // Mock delete
      await new Promise(resolve => setTimeout(resolve, 800));
      
      setCollections(prev => prev.filter(coll => coll.name !== collectionName));
      showSnackbar('Collection deleted successfully', 'success');
    } catch (error) {
      console.error('Error deleting collection:', error);
      showSnackbar('Failed to delete collection', 'error');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleToggleActive = async (collectionName: string, isActive: boolean) => {
    try {
      // In a real app, we would update the active status on the server
      // For now, we'll just update the local state
      setCollections(prev =>
        prev.map(coll =>
          coll.name === collectionName
            ? { ...coll, isActive: !isActive }
            : { ...coll, isActive: false } // Only one active collection at a time
        )
      );
      
      showSnackbar(
        `Collection ${isActive ? 'deactivated' : 'activated'} successfully`,
        'success'
      );
    } catch (error) {
      console.error('Error toggling collection status:', error);
      showSnackbar('Failed to update collection status', 'error');
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Vector Store</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => handleOpenDialog()}
        >
          New Collection
        </Button>
      </Box>

      <Alert severity="info" icon={<InfoIcon />} sx={{ mb: 3 }}>
        Vector stores organize your documents for efficient similarity search. Each collection can contain documents with similar content types or purposes.
      </Alert>

      <StyledPaper>
        <Typography variant="h6" gutterBottom>
          Collections
        </Typography>
        
        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : collections.length === 0 ? (
          <Box textAlign="center" p={4}>
            <StorageIcon color="action" style={{ fontSize: 48, opacity: 0.5 }} />
            <Typography variant="subtitle1" color="textSecondary" gutterBottom>
              No collections found
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Create your first collection to start organizing your documents.
            </Typography>
            <Button
              variant="outlined"
              color="primary"
              startIcon={<AddIcon />}
              onClick={() => handleOpenDialog()}
              sx={{ mt: 2 }}
            >
              Create Collection
            </Button>
          </Box>
        ) : (
          <CollectionList>
            {collections.map((collection) => (
              <React.Fragment key={collection.name}>
                <ListItem>
                  <ListItemIcon>
                    <StorageIcon
                      color={collection.isActive ? 'primary' : 'inherit'}
                    />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center">
                        <Typography
                          variant="subtitle1"
                          sx={{
                            fontWeight: collection.isActive ? 'bold' : 'normal',
                            color: collection.isActive ? 'primary.main' : 'inherit',
                          }}
                        >
                          {collection.name}
                        </Typography>
                        {collection.isActive && (
                          <Tooltip title="Active collection">
                            <CheckCircleIcon
                              color="primary"
                              fontSize="small"
                              sx={{ ml: 1 }}
                            />
                          </Tooltip>
                        )}
                      </Box>
                    }
                    secondary={
                      <>
                        <span>{collection.count.toLocaleString()} documents</span>
                        <br />
                        <span>Updated: {formatDate(collection.updatedAt)}</span>
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    <Tooltip
                      title={
                        collection.isActive
                          ? 'Active collection'
                          : 'Set as active collection'
                      }
                    >
                      <span>
                        <Switch
                          edge="end"
                          checked={collection.isActive}
                          onChange={() =>
                            handleToggleActive(collection.name, collection.isActive)
                          }
                          inputProps={{ 'aria-labelledby': collection.name }}
                        />
                      </span>
                    </Tooltip>
                    <Tooltip title="Edit collection">
                      <IconButton
                        edge="end"
                        aria-label="edit"
                        onClick={() => handleOpenDialog(collection)}
                        sx={{ ml: 1 }}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete collection">
                      <span>
                        <IconButton
                          edge="end"
                          aria-label="delete"
                          onClick={() => handleDeleteCollection(collection.name)}
                          disabled={isDeleting}
                          sx={{ ml: 1 }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </span>
                    </Tooltip>
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider component="li" />
              </React.Fragment>
            ))}
          </CollectionList>
        )}
      </StyledPaper>

      <Dialog open={isDialogOpen} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingCollection ? 'Edit Collection' : 'Create New Collection'}
        </DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            id="name"
            label="Collection Name"
            type="text"
            fullWidth
            variant="outlined"
            value={collectionName}
            onChange={(e) => setCollectionName(e.target.value)}
            placeholder="e.g., genomic_guidelines"
            InputLabelProps={{
              shrink: true,
            }}
            sx={{ mt: 2 }}
          />
          {editingCollection && (
            <Typography variant="caption" color="textSecondary" display="block" mt={2}>
              Created: {formatDate(editingCollection.createdAt)}
              <br />
              Last updated: {formatDate(editingCollection.updatedAt)}
              <br />
              Documents: {editingCollection.count.toLocaleString()}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} color="inherit">
            Cancel
          </Button>
          <Button
            onClick={handleSaveCollection}
            color="primary"
            variant="contained"
            disabled={!collectionName.trim()}
          >
            {editingCollection ? 'Save Changes' : 'Create Collection'}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default VectorStorePage;
