import React, { useState } from 'react';

const FullScreenToggle = () => {
    const [isFullscreen, setIsFullscreen] = useState(false);

    const handleFullscreen = () => {
        if (!isFullscreen) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
        setIsFullscreen(!isFullscreen);
    };

    return (
        <button onClick={handleFullscreen} className="px-4 py-2 bg-green-500 text-white rounded">
            {isFullscreen ? 'Exit Fullscreen' : 'Go Fullscreen'}
        </button>
    );
};

export default FullScreenToggle;
