import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './module/App';
import * as process from 'process';
import { APP_EVENT, APP_PARAMS } from './const';

(window).global = window;
(window).process = process;
(window).Buffer = [];

setInterval(() => {
    window.dispatchEvent(new CustomEvent(APP_EVENT.GAZE));
}, 1000 / APP_PARAMS.FPS);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);