import cv2
import numpy as np
import tensorflow as tf
import math
from face_detector import find_faces
from face_landmarks import detect_marks

model_pts = np.array([
                    (0.0, 0.0, 0.0),
                    (0.0, -330.0, -65.0),
                    (-225.0, 170.0, -135.0),
                    (225.0, 170.0, -135.0),
                    (-150.0, -150.0, -125.0),
                    (150.0, -150.0, -125.0)
                    ])

font = cv2.FONT_HERSHEY_PLAIN
landmark = tf.saved_model.load('models')
face_detector = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/model.caffemodel")

capture = cv2.VideoCapture(0)
available, img = capture.read()
img_size = img.shape

# Camera internals
focal_length = img_size[1]
center = (img_size[1]/2, img_size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

# Control information visibility
show_info = True

def get_points_in_2d(img, rotation_vector, translation_vector, camera_matrix, val):
    rear_depth = val[1]
    rear_size = val[0]
    dist_coeffs = np.zeros((4,1))
    point_3d = []

    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_depth = val[3]
    front_size = val[2]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def get_head_pose_pts(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_points_in_2d(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)

while True:
    available, img = capture.read()
    if available == True:
        faces = find_faces(img, face_detector)
        for face in faces:
            dist_coeffs = np.zeros((4,1))
            marks = detect_marks(img, landmark, face)
            image_pts = np.array([marks[30], marks[8], marks[36], marks[45], marks[48], marks[54]], dtype="double")
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            for p in image_pts:
                p1 = ( int(image_pts[0][0]), int(image_pts[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = get_head_pose_pts(img, rotation_vector, translation_vector, camera_matrix)

                if show_info:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
                    cv2.line(img, p1, p2, (255, 0, 255), 2)
                    cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                a2 = int(math.degrees(math.atan(-1/m)))
            except:
                a2 = 90

            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                a1 = int(math.degrees(math.atan(m)))
            except:
                a1 = 90

            if a1 >= 48 or a1 <= -48 or a2 >= 48 or a2 <= -48:
                print('Turned Away!')
                cv2.putText(img, 'Turned Away!', (30, 30), font, 1, (255, 0, 0), 2)
            else:
                print('Paying Attention!')
                cv2.putText(img, 'Paying Attention!', (30, 30), font, 1, (0, 255, 0), 2)

            if show_info:
                cv2.putText(img, str(a1), tuple(p1), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(a2), tuple(x1), font, 1, (255, 255, 255), 2)
        cv2.imshow('img', img)

        # Show/Hide onscreen overlay
        if cv2.waitKey(1) & 0xFF == ord('s'):
            show_info = not show_info
    else:
        break

capture.release()
cv2.destroyAllWindows()
