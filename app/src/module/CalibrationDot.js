import React from 'react';

const CalibrationDot = ({ position, dotSize }) => {
    return (
        <div className="fixed top-0 left-0 w-full h-full">
            <div
                style={{ transform: `translate(${position[0] - (dotSize / 2)}px, ${position[1] - (dotSize / 2)}px)`, width: dotSize, height: dotSize }}
                className="bg-red-500 rounded-full animate-pulse absolute top-0 left-0 transition-transform duration-1000 linear"
            />
        </div>
    );
}

export default CalibrationDot;
