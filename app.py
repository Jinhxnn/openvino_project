import streamlit as st
import cv2
from utils import load_models, process_frame

def main():
    st.title("Elderly Fall Detection System")

    # Load models
    person_model, pose_model = load_models()

    # Streamlit button to start detection
    if st.button('Start Detection'):
        st.write("Detection Started")
        
        # Open video capture
        cap = cv2.VideoCapture(0)
        alert_threshold = 0.5  # Adjust as needed
        
        # Streamlit video placeholder
        frame_placeholder = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            
            # Process the frame
            processed_frame = process_frame(frame, person_model, pose_model, alert_threshold)
            
            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Show the processed frame
            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

            # Add a stop condition
            if st.button('Stop Detection'):
                st.write("Detection Stopped")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
