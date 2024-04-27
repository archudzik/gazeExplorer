import React, { useEffect, useRef } from 'react';
import Peer from 'simple-peer';
import { APP_ENDPOINT, APP_EVENT } from '../const';

const VideoStream = ({ setGazePointX, setGazePointY, setIsCalibrated }) => {
    const videoRef = useRef();

    const setSrcObject = (newRef) => {
        if (videoRef.current) {
            videoRef.current.srcObject = newRef;
        }
    }

    useEffect(() => {
        navigator.mediaDevices
            .getUserMedia({
                audio: false,
                video: {
                    facingMode: "user",
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            }).then(localStream => {

                let connected = false;
                setSrcObject(localStream);

                const peer = new Peer({
                    initiator: true,
                    trickle: false,
                    streams: [localStream],
                    reconnectTimer: 100,
                    iceTransportPolicy: 'relay',
                });

                peer.on('signal', data => {
                    fetch(APP_ENDPOINT.SIGNAL_SERVER, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data),
                    }).then(response => {
                        return response.json();
                    }).then(serverSignal => {
                        peer.signal(serverSignal);
                    }).catch((reason) => {
                        console.log('Peer connection not available');
                    });
                });

                peer.on('connect', () => {
                    console.log('Peer connection established');
                    connected = true;
                });

                peer.on('open', function (id) {
                    console.log('My peer ID is: ' + id);
                });

                peer.on('error', (err) => {
                    console.error('Peer connection error:', err);
                    connected = false;
                });

                peer.on('data', (data) => {
                    const dataObj = JSON.parse(data) || {};
                    if ('x' in dataObj) {
                        setGazePointX(dataObj.x);
                    }
                    if ('y' in dataObj) {
                        setGazePointY(dataObj.y);
                    }
                    if ('cb' in dataObj) {
                        setIsCalibrated(dataObj.cb);
                    }
                });

                peer.on('stream', (remoteStream) => {
                    setSrcObject(remoteStream);
                });

                window.addEventListener(APP_EVENT.GAZE, (e) => {
                    if (connected) {
                        peer.send('xy');
                    }
                }, false);

                window.addEventListener(APP_EVENT.CALIBRATION, (e) => {
                    if (connected) {
                        const data = JSON.stringify(e.detail);
                        peer.send('cb' + data);
                    }
                }, false);

            }).catch((e) => {
                console.error(e);
            });
    }, []);

    return <video id='video' ref={videoRef} autoPlay className="shadow-lg rounded-lg" />;
};

export default VideoStream;