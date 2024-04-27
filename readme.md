# gazeExplorer

#### Eye and face tracking using React, Python, and WebRTC.

gazeExplorer is an open-source application designed to demonstrate modern eye-tracking capabilities using a client-server architecture. The project uses a thin-client approach, built with React for the frontend and Python for the backend, in order to present how complex computations like gaze detection and salient point detection can be offloaded to a server through WebRTC.

![gazeExplorer](demo.gif)

# Introduction

The primary purpose of this project is to provide an example of web-based eye-tracking technology that is both lightweight on client resources and powerful in computational ability. Implementation of WebRTC allows for robust, real-time communication between the client and server. The React frontend handles capturing webcam images and interfacing with the user, while the Python backend processes these images to detect facial landmarks, calculate gaze vectors, and determine the direction of the face. This information can either be visualized directly on the returned images or provided in raw data (JSON) format for further processing on the client side.

This application can be useful for developers looking to implement eye-tracking functionalities in web applications, especially in the context of assistive technology solutions where access to various data points through a webcam can give possibility to build custom and interactive interfaces.

Therefore, this project is open-sourced in order to support community of developers interested in advancing web-based eye-tracking technologies, providing a scalable and adaptable framework for further development.

# Methods

This section outlines the technical methodologies used in gazeExplorer to achieve real-time gaze tracking through a client-server architecture.

## 1. Webcam Image Capture and Transmission

The React frontend captures the video stream from the user's webcam. This stream is then transmitted to the Python server via the WebRTC protocol, thus with minimal latency and high-efficiency data transfer. This real-time communication is used for maintaining the responsiveness of the gaze and face tracking features.

## 2. Facial Landmark Detection

Upon receiving the video frames, the Python backend uses the dlib library's 68-point face shape predictor to detect facial landmarks. These landmarks are important for identifying key facial features necessary for subsequent processing steps like gaze and head pose estimation.

## 3. Smoothing Facial Landmarks

Given the real-time nature of webcam input, which can often be noisy, an Alpha-Beta Filter is applied to the detected facial landmarks. This smoothing technique helps stabilize the landmarks across consecutive frames, reducing jitter and improving the accuracy of landmark-dependent computations.

## 4. Gaze Direction Estimation

The gaze direction is estimated by analyzing the position of the iris relative to the detected eye landmarks. Here, the Python backend uses the custom trained model for dlib library which is a 16-point eye shape predictor to detect eye landmarks This method provides a directional vector indicating where the user is looking, which is computed for every frame received from the client.

## 5. Head Pose Estimation

The application incorporates head pose estimation using a Perspective-n-Point (PnP) algorithm. This algorithm uses 2D facial landmarks from the video frame and correlates them with a 3D model of a generic human face to compute the rotation and translation vectors. These vectors describe the orientation and position of the head relative to the camera.

## 6. Data Projection and Feedback

Processed data, including gaze vectors and annotated frames with facial landmarks, are sent back to the client. Users can see their gaze direction overlaid on the video feed in real-time. **Alternatively, raw (JSON) data can be sent to the client for further processing or integration into other applications.**

## 7. Multi-Client Handling

The server-side implementation is designed to handle multiple clients simultaneously. Using the MediaRelay class, each session maintains its state independently, allowing the server to manage and process video streams from different users effectively.

## 8. Asynchronous Communication and Session Management

The server handles asynchronous signaling for WebRTC session setup and real-time data communication. How sessions are established and managed is helpful in maintaining the performance and reliability of the application.

## 9. Gaze and Head Pose Estimation Techniques

The server performs basic landmark detection and integrates techniques for gaze estimation and head pose estimation using 3D model points. This involves geometric transformations and camera model adjustments that might be of interest to developers working with computer vision and augmented reality.

## 10. Brightness Detection and Face Detection

The code handles detection in video frame brightness that can be used to optimize face detection, which can be important for applications in diverse lighting conditions.

## 11. Error Handling and Robustness

The implementation includes error handling to ensure the server remains stable and operational even when faced with processing anomalies or network issues.

## 12. Security and Encryption

The optional use of SSL for secure communications is an important aspect, especially for applications handling sensitive data like biometric information.

# Usage

This guide provides step-by-step instructions on setting up and running the **gazeExplorer** application on your local machine. The project structure includes two main directories: `app` for the React frontend and `srv` for the Python backend.

## Setting up the Client (React)

### Prerequisites

- Ensure you have npm installed on your machine.
- A webcam connected to your machine and adequate lighting conditions are necessary for the AI models to function correctly.

### Installation

Navigate to the `app` directory:

```bash
cd app
```

Install the necessary npm packages:

```bash
npm install
```

Start the React application:

```bash
npm start
```

This will launch the frontend on localhost:3000 and will connect to the backend server for processing the webcam data.

## Setting up the Server (Python)

### Prerequisites

- Ensure you have Python 3 installed on your machine.
- The application requires cmake for certain dependencies. Install cmake from cmake.org.

### Installation

Navigate to the srv directory:

```bash
cd srv
```

Install the required Python packages listed in requirements.txt:

```bash
pip install -r requirements.txt
```

### Running the Server

Execute the server script:

```python
python main.py
```

This will start the backend server, which processes incoming video streams from the client using WebRTC and performs gaze tracking and face detection.

## Connecting the Client and Server

- Ensure both the client and server are running. The server will handle incoming connections from the client automatically.
- Access the web application through your browser at localhost:3000 to start using the gaze tracking features.

Please ensure that both client and server are configured to communicate over the same local network or configured IP addresses if not running on the same machine.

# License

The **gazeExplorer** is open-sourced under the MIT License. However, **gazeExplorer** incorporates several dependencies which are covered by their own licenses:

- **OpenCV (cv2)**: BSD license
- **dlib**: Boost Software License
- **NumPy** and **SciPy**: BSD license
- **aiohttp_cors** and **aiortc**: MIT License

We recommend reviewing the licenses of these individual libraries to ensure compliance with their terms when using **gazeExplorer**.

# References

If you find the **gazeExplorer** project useful in your research or if it has inspired your work, please consider citing the following paper to support development:

```
Chudzik, Artur, Albert Śledzianowski, and Andrzej W. Przybyszewski. "Machine Learning and Digital Biomarkers Can Detect Early Stages of Neurodegenerative Diseases." Sensors 24.5 (2024): 1572.
```
