import cv2
import numpy as np
import math
import time

# =============================================================
# CẤU HÌNH CAMERA & CỬA SỔ
# =============================================================
gst_pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=480, height=480, framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture(0) # Dùng cho Webcam laptop

if not cap.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()

def nothing(x): pass

cv2.namedWindow("Control Panel")
cv2.resizeWindow("Control Panel", 400, 350)
cv2.createTrackbar("Line Length", "Control Panel", 100, 100, nothing)
cv2.createTrackbar("Threshold", "Control Panel", 27, 255, nothing)
cv2.createTrackbar("Min Thickness", "Control Panel", 1, 20, nothing)
cv2.createTrackbar("Max Thickness", "Control Panel", 8, 50, nothing)
cv2.createTrackbar("Shadow Buffer", "Control Panel", 10, 30, nothing)
cv2.createTrackbar("Show Text (0/1)", "Control Panel", 1, 1, nothing)

# =============================================================
# BIẾN THỜI GIAN & CẤU HÌNH BÀN CỜ
# =============================================================
last_capture_time = time.time()
capture_interval = 0
display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

ROW_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

print(f"Camera đã chạy! Chụp mỗi {capture_interval} giây.")

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()

    if current_time - last_capture_time >= capture_interval:
        print("\n--- BẮT ĐẦU XỬ LÝ ẢNH MỚI ---")
        
        # 1. Lấy thông số
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        length_val = cv2.getTrackbarPos("Line Length", "Control Panel")
        thresh_val = cv2.getTrackbarPos("Threshold", "Control Panel")
        min_thick = cv2.getTrackbarPos("Min Thickness", "Control Panel")
        max_thick = cv2.getTrackbarPos("Max Thickness", "Control Panel")
        shadow_buf = cv2.getTrackbarPos("Shadow Buffer", "Control Panel")
        show_text = cv2.getTrackbarPos("Show Text (0/1)", "Control Panel")
        
        if length_val < 1: length_val = 1
        if max_thick < min_thick: max_thick = min_thick + 1
        
        # 2. Tìm quân cờ
        pieces_list = []
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1.2, 30,
                                   param1=100, param2=30, minRadius=20, maxRadius=25)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                pieces_list.append((x, y, r))
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

        # 3. TÌM GIAO ĐIỂM & LỌC ĐỘ DÀY
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        scale = max(5, 120 - length_val)
        h_size = int(frame.shape[1] / scale)
        v_size = int(frame.shape[0] / scale)
        
        mask_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1)))
        mask_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size)))

        # --- Lọc độ dày (Thickness Filter) ---
        cnts_h, _ = cv2.findContours(mask_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_h:
            x, y, w, h = cv2.boundingRect(cnt)
            if not (min_thick <= h <= max_thick):
                cv2.drawContours(mask_h, [cnt], -1, 0, -1)

        cnts_v, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_v:
            x, y, w, h = cv2.boundingRect(cnt)
            if not (min_thick <= w <= max_thick):
                cv2.drawContours(mask_v, [cnt], -1, 0, -1)
        
        # Gộp mask
        mask_joints = cv2.bitwise_and(cv2.dilate(mask_h, np.ones((3,3)), iterations=3), 
                                      cv2.dilate(mask_v, np.ones((3,3)), iterations=3))

        contours, _ = cv2.findContours(mask_joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_points = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 0:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    is_shadow = False
                    for (px, py, pr) in pieces_list:
                        if math.sqrt((cx - px)**2 + (cy - py)**2) < (pr + shadow_buf):
                            is_shadow = True; break
                    if not is_shadow:
                        raw_points.append((cx, cy))

        # =============================================================
        # 4. LỌC NHIỄU THEO GRID (ĐIỀU KIỆN CHẶT CHẼ)
        # =============================================================
        final_points = []
        GRID_TOLERANCE = 7 
        MIN_NEIGHBORS = 6  # <--- YÊU CẦU: PHẢI CÓ TỪ 7 ĐIỂM TRỞ LÊN

        for i, (cx, cy) in enumerate(raw_points):
            row_count = 0
            col_count = 0
            
            # Quét toàn bộ các điểm khác để đếm
            for j, (tx, ty) in enumerate(raw_points):
                # Cùng hàng (sai số Y <= 5)
                if abs(cy - ty) <= GRID_TOLERANCE: 
                    row_count += 1
                
                # Cùng cột (sai số X <= 5)
                if abs(cx - tx) <= GRID_TOLERANCE: 
                    col_count += 1
            
            # --- ĐIỀU KIỆN QUAN TRỌNG NHẤT ---
            # Dùng toán tử 'and': Cả 2 điều kiện phải ĐÚNG thì mới giữ lại.
            if row_count >= MIN_NEIGHBORS and col_count >= MIN_NEIGHBORS:
                final_points.append((cx, cy))

        # =============================================================

     # =============================================================
        # 5. CHUYỂN ĐỔI TOẠ ĐỘ ROBOT (GỐC A1 = 0,0)
        # =============================================================
        
        # --- Cấu hình kích thước thật (mm) ---
        REAL_WIDTH_A1_A9  = 230.0  # mm (Chiều ngang)
        REAL_HEIGHT_A1_J1 = 234.0  # mm (Chiều dọc - bao gồm cả sông)

        mapped_points = []
        
        # Biến lưu toạ độ pixel của các điểm mốc để tính tỉ lệ
        pixel_A1 = None
        pixel_A9 = None
        pixel_J1 = None

        if len(final_points) > 10:
            # 1. Gán nhãn A1 -> J9 (Mapping Label)
            # ---------------------------------------------------------
            xs = [p[0] for p in final_points]
            ys = [p[1] for p in final_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y
            
            step_x = width / 8.0
            step_y = height / 9.0
            
            for (cx, cy) in final_points:
                col_idx = int(round((cx - min_x) / step_x))
                row_idx = int(round((cy - min_y) / step_y))
                col_idx = max(0, min(col_idx, 8))
                row_idx = max(0, min(row_idx, 9))
                
                label_str = f"{ROW_LABELS[row_idx]}{col_idx + 1}"
                mapped_points.append({'label': label_str, 'px': cx, 'py': cy})

                # Bắt lấy các điểm mốc
                if label_str == 'A1': pixel_A1 = (cx, cy)
                if label_str == 'A9': pixel_A9 = (cx, cy)
                if label_str == 'J1': pixel_J1 = (cx, cy)
                
                # Vẽ điểm
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # 2. Tính toán tỉ lệ mm/pixel (Scale Factor)
            # ---------------------------------------------------------
            if pixel_A1 and pixel_A9 and pixel_J1:
                # Tính khoảng cách pixel theo trục X (A1 -> A9)
                dist_pixel_x = abs(pixel_A9[0] - pixel_A1[0])
                
                # Tính khoảng cách pixel theo trục Y (A1 -> J1)
                dist_pixel_y = abs(pixel_J1[1] - pixel_A1[1])

                if dist_pixel_x > 0 and dist_pixel_y > 0:
                    scale_x = REAL_WIDTH_A1_A9 / dist_pixel_x   # (mm/pixel)
                    scale_y = REAL_HEIGHT_A1_J1 / dist_pixel_y  # (mm/pixel)
                    
                    print(f"\n--- TÍNH TOÁN TOẠ ĐỘ (Gốc A1=0,0) ---")
                    print(f"Scale X: {scale_x:.4f} mm/px | Scale Y: {scale_y:.4f} mm/px")
                    print(f"{'Điểm':<5} | {'Pixel':<10} | {'Robot (mm)':<15}")
                    print("-" * 45)

                    # Sắp xếp danh sách điểm
                    mapped_points.sort(key=lambda k: k['label'])
                    
                    # 3. Chuyển đổi từng điểm
                    # ---------------------------------------------------------
                    for pt in mapped_points:
                        px, py = pt['px'], pt['py']
                        
                        # CÔNG THỨC QUAN TRỌNG NHẤT:
                        # X_robot = (Pixel_X - Pixel_A1_X) * Scale_X
                        # Y_robot = (Pixel_A1_Y - Pixel_Y) * Scale_Y  <-- Đảo chiều vì Pixel Y ngược Robot Y
                        
                        rob_x = (px - pixel_A1[0]) * scale_x
                        rob_y = (pixel_A1[1] - py) * scale_y 
                        
                        # Làm tròn 1 số thập phân
                        rob_x = round(rob_x, 1)
                        rob_y = round(rob_y, 1)

                        print(f"{pt['label']:<5} | ({px:>3},{py:>3}) | ({rob_x:>5}, {rob_y:>5})")

                        # Hiển thị lên màn hình
                        if show_text == 1:
                            cv2.putText(frame, f"{int(rob_x)},{int(rob_y)}", (px - 20, py + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            # Tô đậm điểm gốc A1
                            if pt['label'] == 'A1':
                                cv2.circle(frame, (px, py), 8, (255, 0, 0), -1) # Màu xanh
                else:
                    print("Lỗi: Khoảng cách pixel quá nhỏ, không thể tính scale.")
            else:
                print(f"Đang tìm điểm mốc... (Đã thấy: A1={bool(pixel_A1)}, A9={bool(pixel_A9)}, J1={bool(pixel_J1)})")
                print("Cần thấy đủ 3 điểm: A1, A9, J1 để tính tỉ lệ.")

        display_frame = frame
        last_capture_time = current_time
# =============================================================
        # 5. CHUYỂN ĐỔI TOẠ ĐỘ ROBOT (GỐC A1 = 0,0)
        # =============================================================
        
        # --- Cấu hình kích thước thật (mm) ---
        REAL_WIDTH_A1_A9  = 230.0  # mm (Chiều ngang)
        REAL_HEIGHT_A1_J1 = 234.0  # mm (Chiều dọc - bao gồm cả sông)

        mapped_points = []
        
        # Biến lưu toạ độ pixel của các điểm mốc để tính tỉ lệ
        pixel_A1 = None
        pixel_A9 = None
        pixel_J1 = None

        if len(final_points) > 10:
            # 1. Gán nhãn A1 -> J9 (Mapping Label)
            # ---------------------------------------------------------
            xs = [p[0] for p in final_points]
            ys = [p[1] for p in final_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y
            
            step_x = width / 8.0
            step_y = height / 9.0
            
            for (cx, cy) in final_points:
                col_idx = int(round((cx - min_x) / step_x))
                row_idx = int(round((cy - min_y) / step_y))
                col_idx = max(0, min(col_idx, 8))
                row_idx = max(0, min(row_idx, 9))
                
                label_str = f"{ROW_LABELS[row_idx]}{col_idx + 1}"
                mapped_points.append({'label': label_str, 'px': cx, 'py': cy})

                # Bắt lấy các điểm mốc
                if label_str == 'A1': pixel_A1 = (cx, cy)
                if label_str == 'A9': pixel_A9 = (cx, cy)
                if label_str == 'J1': pixel_J1 = (cx, cy)
                
                # Vẽ điểm
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # 2. Tính toán tỉ lệ mm/pixel (Scale Factor)
            # ---------------------------------------------------------
            if pixel_A1 and pixel_A9 and pixel_J1:
                # Tính khoảng cách pixel theo trục X (A1 -> A9)
                dist_pixel_x = abs(pixel_A9[0] - pixel_A1[0])
                
                # Tính khoảng cách pixel theo trục Y (A1 -> J1)
                dist_pixel_y = abs(pixel_J1[1] - pixel_A1[1])

                if dist_pixel_x > 0 and dist_pixel_y > 0:
                    scale_x = REAL_WIDTH_A1_A9 / dist_pixel_x   # (mm/pixel)
                    scale_y = REAL_HEIGHT_A1_J1 / dist_pixel_y  # (mm/pixel)
                    
                    print(f"\n--- TÍNH TOÁN TOẠ ĐỘ (Gốc A1=0,0) ---")
                    print(f"Scale X: {scale_x:.4f} mm/px | Scale Y: {scale_y:.4f} mm/px")
                    print(f"{'Điểm':<5} | {'Pixel':<10} | {'Robot (mm)':<15}")
                    print("-" * 45)

                    # Sắp xếp danh sách điểm
                    mapped_points.sort(key=lambda k: k['label'])
                    
                    # 3. Chuyển đổi từng điểm
                    # ---------------------------------------------------------
                    for pt in mapped_points:
                        px, py = pt['px'], pt['py']
                        
                        # CÔNG THỨC QUAN TRỌNG NHẤT:
                        # X_robot = (Pixel_X - Pixel_A1_X) * Scale_X
                        # Y_robot = (Pixel_A1_Y - Pixel_Y) * Scale_Y  <-- Đảo chiều vì Pixel Y ngược Robot Y
                        
                        rob_x = (px - pixel_A1[0]) * scale_x
                        rob_y = (pixel_A1[1] - py) * scale_y 
                        
                        # Làm tròn 1 số thập phân
                        rob_x = round(rob_x, 1)
                        rob_y = round(rob_y, 1)

                        print(f"{pt['label']:<5} | ({px:>3},{py:>3}) | ({rob_x:>5}, {rob_y:>5})")

                        # Hiển thị lên màn hình
                        if show_text == 1:
                            cv2.putText(frame, f"{int(rob_x)},{int(rob_y)}", (px - 20, py + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            # Tô đậm điểm gốc A1
                            if pt['label'] == 'A1':
                                cv2.circle(frame, (px, py), 8, (255, 0, 0), -1) # Màu xanh
                else:
                    print("Lỗi: Khoảng cách pixel quá nhỏ, không thể tính scale.")
            else:
                print(f"Đang tìm điểm mốc... (Đã thấy: A1={bool(pixel_A1)}, A9={bool(pixel_A9)}, J1={bool(pixel_J1)})")
                print("Cần thấy đủ 3 điểm: A1, A9, J1 để tính tỉ lệ.")

        display_frame = frame
        last_capture_time = current_time
# =============================================================
        # 5. CHUYỂN ĐỔI TOẠ ĐỘ ROBOT (GỐC A1 = 0,0)
        # =============================================================
        
        # --- Cấu hình kích thước thật (mm) ---
        REAL_WIDTH_A1_A9  = 230.0  # mm (Chiều ngang)
        REAL_HEIGHT_A1_J1 = 234.0  # mm (Chiều dọc - bao gồm cả sông)

        mapped_points = []
        
        # Biến lưu toạ độ pixel của các điểm mốc để tính tỉ lệ
        pixel_A1 = None
        pixel_A9 = None
        pixel_J1 = None

        if len(final_points) > 10:
            # 1. Gán nhãn A1 -> J9 (Mapping Label)
            # ---------------------------------------------------------
            xs = [p[0] for p in final_points]
            ys = [p[1] for p in final_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y
            
            step_x = width / 8.0
            step_y = height / 9.0
            
            for (cx, cy) in final_points:
                col_idx = int(round((cx - min_x) / step_x))
                row_idx = int(round((cy - min_y) / step_y))
                col_idx = max(0, min(col_idx, 8))
                row_idx = max(0, min(row_idx, 9))
                
                label_str = f"{ROW_LABELS[row_idx]}{col_idx + 1}"
                mapped_points.append({'label': label_str, 'px': cx, 'py': cy})

                # Bắt lấy các điểm mốc
                if label_str == 'A1': pixel_A1 = (cx, cy)
                if label_str == 'A9': pixel_A9 = (cx, cy)
                if label_str == 'J1': pixel_J1 = (cx, cy)
                
                # Vẽ điểm
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # 2. Tính toán tỉ lệ mm/pixel (Scale Factor)
            # ---------------------------------------------------------
            if pixel_A1 and pixel_A9 and pixel_J1:
                # Tính khoảng cách pixel theo trục X (A1 -> A9)
                dist_pixel_x = abs(pixel_A9[0] - pixel_A1[0])
                
                # Tính khoảng cách pixel theo trục Y (A1 -> J1)
                dist_pixel_y = abs(pixel_J1[1] - pixel_A1[1])

                if dist_pixel_x > 0 and dist_pixel_y > 0:
                    scale_x = REAL_WIDTH_A1_A9 / dist_pixel_x   # (mm/pixel)
                    scale_y = REAL_HEIGHT_A1_J1 / dist_pixel_y  # (mm/pixel)
                    
                    print(f"\n--- TÍNH TOÁN TOẠ ĐỘ (Gốc A1=0,0) ---")
                    print(f"Scale X: {scale_x:.4f} mm/px | Scale Y: {scale_y:.4f} mm/px")
                    print(f"{'Điểm':<5} | {'Pixel':<10} | {'Robot (mm)':<15}")
                    print("-" * 45)

                    # Sắp xếp danh sách điểm
                    mapped_points.sort(key=lambda k: k['label'])
                    
                    # 3. Chuyển đổi từng điểm
                    # ---------------------------------------------------------
                    for pt in mapped_points:
                        px, py = pt['px'], pt['py']
                        
                        # CÔNG THỨC QUAN TRỌNG NHẤT:
                        # X_robot = (Pixel_X - Pixel_A1_X) * Scale_X
                        # Y_robot = (Pixel_A1_Y - Pixel_Y) * Scale_Y  <-- Đảo chiều vì Pixel Y ngược Robot Y
                        
                        rob_x = (px - pixel_A1[0]) * scale_x
                        rob_y = (pixel_A1[1] - py) * scale_y 
                        
                        # Làm tròn 1 số thập phân
                        rob_x = round(rob_x, 1)
                        rob_y = round(rob_y, 1)

                        print(f"{pt['label']:<5} | ({px:>3},{py:>3}) | ({rob_x:>5}, {rob_y:>5})")

                        # Hiển thị lên màn hình
                        if show_text == 1:
                            cv2.putText(frame, f"{int(rob_x)},{int(rob_y)}", (px - 20, py + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            # Tô đậm điểm gốc A1
                            if pt['label'] == 'A1':
                                cv2.circle(frame, (px, py), 8, (255, 0, 0), -1) # Màu xanh
                else:
                    print("Lỗi: Khoảng cách pixel quá nhỏ, không thể tính scale.")
            else:
                print(f"Đang tìm điểm mốc... (Đã thấy: A1={bool(pixel_A1)}, A9={bool(pixel_A9)}, J1={bool(pixel_J1)})")
                print("Cần thấy đủ 3 điểm: A1, A9, J1 để tính tỉ lệ.")

        display_frame = frame
        last_capture_time = current_time





    cv2.imshow("Ket qua (Static Update 1s)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




