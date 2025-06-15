import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    primary: {
      main: '#005eb8', // NHS Blue
      light: '#41b6e6', // NHS Light Blue
      dark: '#003087', // NHS Dark Blue
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#007f3b', // NHS Green
      light: '#78be20', // NHS Light Green
      dark: '#00401d',
      contrastText: '#ffffff',
    },
    error: {
      main: '#da291c', // NHS Red
    },
    warning: {
      main: '#ffeb3b', // NHS Yellow
    },
    background: {
      default: '#f0f4f5',
      paper: '#ffffff',
    },
    text: {
      primary: '#212b32', // NHS Black
      secondary: '#425563', // NHS Dark Grey
    },
  },
  typography: {
    fontFamily: '"Frutiger W01", Arial, sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          padding: '8px 16px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.05)',
        },
      },
    },
  },
});

export default theme;
