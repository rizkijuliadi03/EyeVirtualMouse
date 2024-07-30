import cv2
import mediapipe as mp
import pyautogui

# Initialize video capture and MediaPipe Face Mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()


def main():
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()
        if not ret:
            break

        # Flip and convert frame to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Face Mesh
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        # Check if landmarks are detected
        if landmark_points:
            landmarks = landmark_points[0].landmark

            # Draw landmarks for eye control
            for idx, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))

                if idx == 1:
                    # Control mouse movement
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)

            # Draw landmarks for click detection
            left_eye_landmarks = [landmarks[145], landmarks[159]]
            for landmark in left_eye_landmarks:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))

            # Check if eyes are in a close position to trigger a click
            if (left_eye_landmarks[0].y - left_eye_landmarks[1].y) < 0.004:
                pyautogui.click()
                pyautogui.sleep(1)

        # Display the frame
        cv2.imshow('Eye Controlled Mouse', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
