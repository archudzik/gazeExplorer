import React, { useState, useEffect } from 'react';
import VideoStream from './VideoStream';
import FullScreenToggle from './FullScreenToggle';
import CalibrationDot from './CalibrationDot';
import { APP_PARAMS, APP_STAGE } from '../const';

function App() {

    const [currentStage, setCurrentStage] = useState(APP_STAGE.DEFAULT)
    const [currentIndex, setCurrentIndex] = useState(0);
    const [dotSize, setDotSize] = useState(50);
    const [gazePointX, setGazePointX] = useState(-1);
    const [gazePointY, setGazePointY] = useState(-1);
    const [isCalibrated, setIsCalibrated] = useState(false);

    const [calibrationPoints, setCalibrationPoints] = useState([
        [dotSize, dotSize],
        [window.innerWidth / 2, dotSize],
        [window.innerWidth - dotSize, dotSize],
        [dotSize, window.innerHeight / 2],
        [window.innerWidth / 2, window.innerHeight / 2],
        [window.innerWidth - dotSize, window.innerHeight / 2],
        [dotSize, window.innerHeight - dotSize],
        [window.innerWidth / 2, window.innerHeight - dotSize],
        [window.innerWidth - dotSize, window.innerHeight - dotSize],
    ])

    useEffect(() => {
        function handleResize() {
            setCalibrationPoints([
                [dotSize, dotSize],
                [window.innerWidth / 2, dotSize],
                [window.innerWidth - dotSize, dotSize],
                [dotSize, window.innerHeight / 2],
                [window.innerWidth / 2, window.innerHeight / 2],
                [window.innerWidth - dotSize, window.innerHeight / 2],
                [dotSize, window.innerHeight - dotSize],
                [window.innerWidth / 2, window.innerHeight - dotSize],
                [window.innerWidth - dotSize, window.innerHeight - dotSize],
            ])
        }

        // Add event listener for window resize
        window.addEventListener('resize', handleResize)

        // Remove event listener on cleanup
        return () => window.removeEventListener('resize', handleResize)
    }, [dotSize]) // Empty dependency array ensures this runs on mount and unmount only

    const startCalibration = async () => {
        let gazeData = [];
        let calibData = [];
        setCurrentStage(APP_STAGE.CALIBRATION);

        const recordData = (index) => new Promise(resolve => {
            let localCount = 0;
            const requiredCount = 0.5 * (1000 / APP_PARAMS.FPS); // get data for 0,5 second
            const aggInterval = setInterval(() => {
                calibData.push([calibrationPoints[index][0], calibrationPoints[index][1]]);
                gazeData.push([gazePointX, gazePointY]);
                localCount++;
                if (localCount > requiredCount) {
                    clearInterval(aggInterval);
                    resolve();
                }
            }, 1000 / APP_PARAMS.FPS);
        });

        for (let index = 0; index < calibrationPoints.length; index++) {
            setCurrentIndex(index);
            await new Promise(r => setTimeout(r, 1000)); // wait until animation is done
            await recordData(index);
        }

        setCurrentIndex(0);
        setCurrentStage(APP_STAGE.DEFAULT);
    };

    const calibrationView = (
        <CalibrationDot position={calibrationPoints[currentIndex]} dotSize={dotSize} setDotSize={setDotSize} />
    )

    const videoView = (
        <VideoStream setGazePointX={setGazePointX} setGazePointY={setGazePointY} setIsCalibrated={setIsCalibrated} />
    )

    const toolbarView = (
        <div className="absolute bottom-4 right-4 grid grid-cols-2 gap-2">
            <FullScreenToggle />
            <button onClick={() => { startCalibration() }} className="px-4 py-2 bg-blue-500 text-white rounded">Start Calibration</button>
        </div>
    )

    return (
        <>
            <div className="flex items-center justify-center min-h-screen w-full">
                {currentStage === APP_STAGE.DEFAULT ? videoView : null}
                {currentStage === APP_STAGE.DEFAULT ? toolbarView : null}
                {currentStage === APP_STAGE.CALIBRATION ? calibrationView : null}
            </div>
        </>

    );
}

export default App;
