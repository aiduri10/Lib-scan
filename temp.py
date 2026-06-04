import pyrealsense2 as rs
import numpy as np
import cv2
from pyzbar import pyzbar

def main():
    # 1. RealSense 파이프라인(카메라 설정) 초기화
    pipeline = rs.pipeline()
    config = rs.config()

    # 바코드 인식에는 컬러(RGB) 화면만 필요하므로 컬러 스트림만 활성화합니다.
    # 해상도 640x480, 초당 30프레임 설정
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 카메라 스트리밍 시작
    pipeline.start(config)
    print("카메라가 켜졌습니다. 바코드를 비춰보세요. (종료하려면 'q'를 누르세요)")

    try:
        while True:
            # 2. 카메라에서 프레임 받아오기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # RealSense 프레임 데이터를 OpenCV에서 쓸 수 있는 numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())

            # 3. 바코드/QR코드 해독 (pyzbar 활용)
            decoded_objects = pyzbar.decode(color_image)

            for obj in decoded_objects:
                # 인식된 바코드 주변에 사각형 그리기
                points = obj.polygon
                if len(points) == 4:
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(color_image, [pts], True, (0, 255, 0), 3)

                # 바코드 데이터(글자) 및 종류(바코드/QR) 추출
                barcode_data = obj.data.decode("utf-8")
                barcode_type = obj.type
                
                # 화면에 인식된 텍스트 띄우기
                text = f"{barcode_type}: {barcode_data}"
                cv2.putText(color_image, text, (obj.rect.left, obj.rect.top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 터미널에도 결과 출력
                print(f"인식 성공! -> {text}")

            # 4. 화면 출력
            cv2.imshow('RealSense D415 Barcode Reader', color_image)

            # 'q' 키를 누르면 루프 탈출
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 5. 종료 시 카메라 및 창 닫기
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()