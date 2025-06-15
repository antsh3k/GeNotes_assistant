import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  IconButton,
  useTheme,
} from '@mui/material';
import MenuBookIcon from '@mui/icons-material/MenuBook';

const Navbar: React.FC = () => {
  const theme = useTheme();
  
  return (
    <AppBar position="static" elevation={0}>
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <MenuBookIcon sx={{ display: { xs: 'none', md: 'flex' }, mr: 1 }} />
          <Typography
            variant="h6"
            noWrap
            component={RouterLink}
            to="/"
            sx={{
              mr: 2,
              display: { xs: 'none', md: 'flex' },
              fontWeight: 700,
              color: 'white',
              textDecoration: 'none',
            }}
          >
            GeNotes
          </Typography>

          <Box sx={{ flexGrow: 1, display: 'flex', ml: 4 }}>
            <Button
              component={RouterLink}
              to="/"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              Chat
            </Button>
            <Button
              component={RouterLink}
              to="/data"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              Data Management
            </Button>
            <Button
              component={RouterLink}
              to="/vector-store"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              Vector Store
            </Button>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Navbar;
