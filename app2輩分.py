from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, jsonify
import pygame

app = Flask(__name__)

previous_elbow_angle_left = None
previous_elbow_angle_right = None
left_punch_counter = [0]
right_punch_counter = [0]


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_dist(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)
    return dist


def detect_punch(elbow_angle, previous_elbow_angle, counter, leg_angle, arm_type):
    if (
        previous_elbow_angle is not None
        and 50 <= previous_elbow_angle < 150
        and elbow_angle < 50
        and 172 < leg_angle
    ):
        counter[0] += 1
        print(f"{arm_type} Punch detected! Counter:", counter[0])
        punch_sound.play()
        print("punch audio played")
        punch_boolean = True


total_duration_rounded = {
    "punching": 0,
    "sitting": 0,
    "squating": 0,
    "running": 0,
    "standing": 0,
    "left punches": 0,
    "right punches": 0,
    "T Pose": 0,
    "Tree Pose": 0,
    "Bridge pose": 0,
    "Downward Facing Dog": 0,
}
remaining_time = 0


pygame.init()
pygame.mixer.init()

punch_sound = pygame.mixer.Sound("punch.mp3")
yoga_sound = pygame.mixer.Sound("yoga2.MP3")
h_sound = pygame.mixer.Sound("here.mp3")
pygame.mixer.music.load("MAIN.mp3")

t_sound = pygame.mixer.Sound("3.mp3")


label = "Unknown Pose"

import socket


def get_ip_address():
    # 獲取本機名稱

    return ip_address


def generate_frames():
    h_sound.play()

    print("1")
    A = round(time.time() * 1000)
    global total_duration
    global remaining_time
    cap = cv2.VideoCapture(0)  # 跑了五秒
    hostname = socket.gethostname()
    # 通過本機名稱獲取本機IP地址
    ip_address = socket.gethostbyname(hostname)
    print("ip_address", ip_address)
    # cap = cv2.VideoCapture("http://admin:1234@192.168.56.1/video")
    B = round(time.time() * 1000)
    print(B - A)
    print(time.ctime())
    frame_width = int(1080)
    print(frame_width)
    frame_height = int(720)
    print(frame_height)

    pygame.mixer.music.play(-1)  # -1 表示无限循环播放

    print("3")
    print(time.ctime())
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    stage = None
    counter = 0
    start_time = 0
    end_time = 0
    total_duration = {
        "punching": 0,
        "sitting": 0,
        "squating": 0,
        "running": 0,
        "standing": 0,
        "left punches": 0,
        "right punches": 0,
        "T Pose": 0,
        "Tree Pose": 0,
        "Bridge pose": 0,
        "Downward Facing Dog": 0,
    }
    start_time = time.time()
    stage_start_time = start_time
    record_duration = 150  # s 秒
    global previous_elbow_angle_left, previous_elbow_angle_right

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:

        while True:
            success, frame = cap.read()
            if not success:
                break
            remaining_time = record_duration - (time.time() - start_time)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # -----------
            try:
                landmarks = results.pose_landmarks.landmark

                lshoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                lelbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                lwrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]
                lhip = [
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                ]
                lankle = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                ]
                lknee = [
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                ]

                rshoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                ]
                relbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                ]
                rwrist = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                ]
                rhip = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                ]
                rankle = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                ]
                rknee = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                ]

                langle = calculate_angle(lshoulder, lelbow, lwrist)
                rangle = calculate_angle(rshoulder, relbow, rwrist)
                lsangle = calculate_angle(lhip, lshoulder, lelbow)
                rsangle = calculate_angle(rhip, rshoulder, relbow)
                ankdist = calculate_dist(lankle, rankle)
                rwdist = calculate_dist(rhip, rwrist)
                lwdist = calculate_dist(lhip, lwrist)
                rhangle = calculate_angle(rshoulder, rhip, rknee)
                lhangle = calculate_angle(lshoulder, lhip, lknee)
                rkangle = calculate_angle(rankle, rknee, rhip)
                lkangle = calculate_angle(lankle, lknee, lhip)
                r_angle = calculate_angle(rhip, rknee, rankle)
                l_angle = calculate_angle(lhip, lknee, lankle)

                hip = (
                    int(
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
                        * frame.shape[1]
                    ),
                    int(
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                        * frame.shape[0]
                    ),
                )
                knee = (
                    int(
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
                        * frame.shape[1]
                    ),
                    int(
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                        * frame.shape[0]
                    ),
                )
                ankle = (
                    int(
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
                        * frame.shape[1]
                    ),
                    int(
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                        * frame.shape[0]
                    ),
                )
                leg_angle = calculate_angle(hip, knee, ankle)

                points = [
                    "LEFT_SHOULDER",
                    "LEFT_ELBOW",
                    "LEFT_WRIST",
                    "RIGHT_SHOULDER",
                    "RIGHT_ELBOW",
                    "RIGHT_WRIST",
                ]
                coords = {
                    point: [
                        landmarks[getattr(mp_pose.PoseLandmark, point).value].x,
                        landmarks[getattr(mp_pose.PoseLandmark, point).value].y,
                    ]
                    for point in points
                }

                elbow_angle_left = calculate_angle(
                    coords["LEFT_SHOULDER"], coords["LEFT_ELBOW"], coords["LEFT_WRIST"]
                )
                elbow_angle_right = calculate_angle(
                    coords["RIGHT_SHOULDER"],
                    coords["RIGHT_ELBOW"],
                    coords["RIGHT_WRIST"],
                )
                punch_boolean = False
                detect_punch(
                    elbow_angle_left,
                    previous_elbow_angle_left,
                    left_punch_counter,
                    leg_angle,
                    "Left",
                )
                detect_punch(
                    elbow_angle_right,
                    previous_elbow_angle_right,
                    right_punch_counter,
                    leg_angle,
                    "Right",
                )

                previous_elbow_angle_left = elbow_angle_left

                previous_elbow_angle_right = elbow_angle_right

                # Check if it is the warrior II pose.
                # ----------------------------------------------------------------------------------------------------------------

                cv2.putText(
                    image,
                    f"Left Punch Count: {left_punch_counter[0]}",
                    (10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Right Punch Count: {right_punch_counter[0]}",
                    (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                """  
                cv2.putText(
                    image,
                    f"Left Elbow: {elbow_angle_left:.2f}°",
                    (300, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            
                cv2.putText(
                    image,
                    (
                        f"Prev Left Elbow: {previous_elbow_angle_left:.2f}°"
                        if previous_elbow_angle_left
                        else "Prev Left Elbow: N/A"
                    ),
                    (300, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                              cv2.putText(
                    frame,
                    f"Left Punch Count: {left_punch_counter[0]}",
                    (0, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Right Punch Count: {right_punch_counter[0]}",
                    (250, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
"""
                foot_on_thigh = (lankle[1] > rhip[1] and lankle[1] < rknee[1]) or (
                    rankle[1] > lhip[1] and rankle[1] < lknee[1]
                )
                # Condition to check if the standing leg is straight
                balance_leg_straight = (
                    abs(r_angle - 180) < 20 or abs(l_angle - 180) < 20
                )

                if (
                    foot_on_thigh
                    and balance_leg_straight
                    and calculate_dist(rwrist, lwrist) < 0.1
                ):
                    stage = "Tree Pose"
                    yoga_sound.play()
                    pygame.mixer.music.pause()

                elif (
                    (previous_elbow_angle_left < 50)
                    or (previous_elbow_angle_right < 50)
                ) and leg_angle > 170:
                    stage = "punching"

                    yoga_sound.stop()

                    pygame.mixer.music.unpause()

                elif ((32 < lsangle < 45)) and (82 > lkangle > 60):
                    stage = "Bridge pose"
                    yoga_sound.play()
                    pygame.mixer.music.pause()

                elif (
                    (abs(langle - 180) < 20 and abs(rangle - 180) < 60)
                    and (abs(lkangle - 180) < 30 and abs(rkangle - 180) < 30)
                    and (
                        lhangle > 60 and lhangle < 83 and rhangle > 60 and rhangle < 83
                    )
                ):
                    stage = "Downward Facing Dog"
                    yoga_sound.play()
                    pygame.mixer.music.pause()

                elif (
                    (rhangle > 80 and lhangle > 80)
                    and (rhangle < 110 and lhangle < 110)
                    and (lkangle < 100 and rkangle < 100)
                ):
                    stage = "sitting"
                    yoga_sound.stop()
                    pygame.mixer.music.unpause()

                elif (10 < r_angle and r_angle < 30) and (
                    10 < l_angle and l_angle < 30
                ):
                    stage = "squating"
                    yoga_sound.stop()
                    pygame.mixer.music.unpause()

                elif 140 < leg_angle < 160:
                    stage = "running"
                    yoga_sound.stop()
                    pygame.mixer.music.unpause()

                elif (abs(langle - 180) < 20 and abs(rangle - 180) < 20) and (
                    abs(lwrist[1] - lshoulder[1]) < 0.1
                    and abs(rwrist[1] - rshoulder[1]) < 0.1
                ):

                    # Specify the label of the pose that is tree pose.
                    stage = "T Pose"
                    yoga_sound.play()
                    pygame.mixer.music.pause()

                elif leg_angle > 175:
                    stage = "standing"
                    yoga_sound.stop()
                    pygame.mixer.music.unpause()

                current_time = time.time()
                stage_duration = current_time - stage_start_time
                total_duration[stage] += stage_duration
                stage_start_time = current_time

            except:
                pass
            cv2.rectangle(image, (0, 0), (420, 80), (240, 230, 140), -1)
            cv2.putText(
                image,
                "STAGE",
                (65, 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if stage == "Downward Facing Dog":
                font_scale = 1  # 如果stage是"10"，使用更大的字体大小
            else:
                font_scale = 2  # 默认字体大小

            # Corrected and properly formatted cv2.putText call to display angle values
            """
            cv2.putText(
                image,  # Image where text is to be added
                str(lsangle) + "  " + str(lkangle),  # Left hip angle  # Right hip angle
                (150, 300),  # Position on the image to start the text
                cv2.FONT_HERSHEY_COMPLEX,  # Font type
                0.5,  # Font size
                (255, 255, 255),  # Font color in BGR
                1,  # Font thickness
                cv2.LINE_AA,  # Line type
            )
"""
            cv2.putText(
                image,
                stage,
                (60, 60),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if 2.9 < remaining_time < 3:

                pygame.mixer.music.stop()
                t_sound.play()
            if remaining_time <= 0:
                total_duration["left punches"] = left_punch_counter[
                    0
                ]  # 將左拳數量添加到 total_duration
                total_duration["right punches"] = right_punch_counter[
                    0
                ]  # 將右拳數量添加到 total_duration
                break  # 結束循環

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2
                ),
            )

            ret, buffer = cv2.imencode(".jpg", image)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    cap.release()

    print(left_punch_counter[0])
    print(right_punch_counter[0])


@app.route("/video_feed")
def video_feed():
    print("now")
    print(time.ctime())

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/duration")
def duration():
    total_duration_rounded = {
        key: f"{round(value)}      " if idx > 2 else f"{round(value)}    "
        for idx, (key, value) in enumerate(total_duration.items())
    }
    print(jsonify(total_duration_rounded))
    return jsonify(total_duration_rounded)


@app.route("/remaining_time")
def get_remaining_time():
    global remaining_time
    remaining_time = round(remaining_time)
    return jsonify({"remaining_time": remaining_time})


@app.route("/audio_feed")
def audio_feed():
    return app.send_from_directory("static", "test.MP3")


@app.route("/")
def index():
    return render_template("run.html")


if __name__ == "__main__":

    print("now")
    print(time.ctime())
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
