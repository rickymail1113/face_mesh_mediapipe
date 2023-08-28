import cv2
import mediapipe as mp

mp_drawng = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawng.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5,
min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        if 27 == 0xFF & cv2.waitKey(1):
            break

        success, image = cap.read()
        if not success:
            print("no video frame")
            continue

        result = face_mesh.process(image)
        if result.multi_face_landmarks:

            for i in result.multi_face_landmarks:
            # 468 個臉部標記
                for point in enumerate(i.landmark):
                    print(point)

                # 臉部 網格
                mp_drawng.draw_landmarks(image=image, landmark_list=i, connections=mp_face_mesh.FACEMESH_TESSELATION,
                                         landmark_drawing_spec=None,
                                         connection_drawing_spec=mp_drawing_style.get_default_face_mesh_tesselation_style())

                # 臉、眉、嘴 輪廓
                mp_drawng.draw_landmarks(image=image, landmark_list=i, connections=mp_face_mesh.FACEMESH_CONTOURS,
                                         landmark_drawing_spec=None,
                                         connection_drawing_spec=mp_drawing_style.get_default_face_mesh_tesselation_style())

                # 瞳孔
                mp_drawng.draw_landmarks(image=image, landmark_list=i, connections=mp_face_mesh.FACEMESH_IRISES,
                                         landmark_drawing_spec=None,
                                         connection_drawing_spec=mp_drawing_style.get_default_face_mesh_tesselation_style())

                cv2.imshow("Face_Mesh", image)

cap.release()
cv2.destroyAllWindows()