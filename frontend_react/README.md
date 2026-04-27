# GRACE Downscaling Engine - React Frontend

A modern, sci-fi themed React UI for the GRACE Downscaling Engine. This frontend replicates all the aesthetics and functionalities of the original Streamlit application.

## Features

✨ **Sci-Fi Aesthetics**
- Neon cyan (#00E5FF) and magenta (#FF00FF) color scheme
- Monospace fonts and glow effects
- Terminal-style components
- Smooth animations and transitions

📱 **Responsive Design**
- Mobile-friendly layout
- Adaptive grid systems
- Touch-optimized controls

🎨 **Complete UI Recreation**
- **Tab 1**: Setup & Area of Interest with interactive inputs and globe placeholder
- **Tab 2**: Data Processing with terminal-style feedback
- **Tab 3**: Model Training configuration and progress tracking
- **Tab 4**: Custom map creation controls
- **Tab 5**: Statistical Analysis with performance metrics

## Installation

```bash
# 1. Install dependencies
npm install

# 2. Configure backend API URL
cp .env.example .env
# Edit .env if your backend is not on http://localhost:5321

# 3. Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

**Note:** The `.env` file is git-ignored. Copy from `.env.example` and configure your backend API URL (default: http://localhost:5321).

## Project Structure

```
frontend_react/
├── src/
│   ├── components/
│   │   ├── Header.jsx & Header.css
│   │   ├── Ticker.jsx & Ticker.css
│   │   ├── SystemMetrics.jsx & SystemMetrics.css
│   │   ├── TabNavigation.jsx & TabNavigation.css
│   │   └── tabs/
│   │       ├── Tab1Setup.jsx & Tab1Setup.css
│   │       ├── Tab2DataProcessing.jsx & Tab2DataProcessing.css
│   │       ├── Tab3ModelTraining.jsx & Tab3ModelTraining.css
│   │       ├── Tab4Maps.jsx & Tab4Maps.css
│   │       └── Tab5Analysis.jsx & Tab5Analysis.css
│   ├── App.jsx & App.css
│   ├── index.css
│   └── main.jsx
├── index.html
├── package.json
└── vite.config.js
```

## Technology Stack

- **React 18.2.0** - UI Framework
- **Vite** - Build tool & dev server
- **CSS3** - Styling with custom animations
- **ES6+** - Modern JavaScript

## Color Palette

- **Primary Cyan**: `#00E5FF`
- **Primary Magenta**: `#FF00FF`
- **Dark Background**: `#0E1117`
- **Muted Text**: `#A0AEC0`
- **Success Green**: `#27c93f`

## Customization

All components use CSS variables at the `:root` level. You can customize colors and animations in:
- `src/index.css` - Global styles
- `src/App.css` - App-level styles
- Individual component CSS files

## Future Enhancements

- [ ] Integrate with backend API
- [ ] Add interactive 3D globe component (Cesium.js or Mapbox)
- [ ] Implement real terminal output streaming
- [ ] Add Plotly.js charts for data visualization
- [ ] State management with Redux or Context API
- [ ] Dark/Light theme toggle
- [ ] Multi-language support

## Notes

This is a UI-first replica of the Streamlit application. To connect it to the backend:

1. Update API endpoints in component files
2. Implement HTTP requests with `axios`
3. Add proper error handling and loading states
4. Integrate session state management

## License

MIT

## Credits

Engineered & Designed by:
- **NEVERSPLIT**
- **ANASTRIA-LAB**
