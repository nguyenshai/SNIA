# SNIA - Swarm and Nature-Inspired Algorithms

Đây là một khuôn khổ (framework) tối ưu hóa toàn diện bằng Python, triển khai nhiều thuật toán tối ưu hóa kinh điển, lấy cảm hứng từ tự nhiên (Sinh học, Vật lý, Tiến hóa) để giải quyết các bài toán Liên tục (Continuous) và Rời rạc (Discrete).

## 📂 Tổ chức mã nguồn (Project Structure)

Dự án được chia thành các module rõ ràng:

- **`algorithms/`**: Chứa các thuật toán giải quyết bài toán.
  - `biology/`: Thuật toán sinh học (ACO, ABC, CS, PSO, FA).
  - `evolution/`: Thuật toán tiến hóa (GA, DE).
  - `physics/`: Thuật toán vật lý (SA).
  - `human/`: Thuật toán xã hội loài người (TLBO).
  - `classical/`: Thuật toán tìm kiếm kinh điển (BFS, DFS, A*, Hill Climbing).
- **`problems/`**: Định nghĩa các bài toán cần tối ưu.
  - `continuous/`: Benchmark liên tục (Sphere, Rastrigin, Ackley, Rosenbrock, Griewank).
  - `discrete/`: Rời rạc / tổ hợp (TSP, Knapsack, Graph Coloring, Shortest Path).
- **`visualization/`**: Chứa các đoạn mã phụ trách vẽ đồ thị (Plotting) nâng cao cho từng dạng bài toán.
  - Vẽ real-time hoạt ảnh (Animation) khi chạy thuật toán, xuất các hình ảnh biểu diễn đường đi của kiến trong TSP, cách đổ màu trong Graph Coloring, hay đồ thị hộp Knapsack.
- **`scripts/`**: Chứa các tệp mã phân tích chuyên sâu (Độ ổn định, Thời gian, Tuning tham số) và sinh ảnh động (GIFs).
- **`results/`**: Nơi lưu trữ toàn bộ file hình ảnh Plot, Boxplot, Heatmap từ các script chạy.
- **`main.py`**: 🔥 **File tiện ích chính (CLI)** dùng để kết nối Thuật toán và Bài toán, thiết lập tham số để chạy nhanh 1 thử nghiệm.

---

## 🚀 1. Giải thích và Cách dùng `main.py`

`main.py` đóng vai trò là điểm vào (Entry point) cung cấp giao diện dòng lệnh (CLI). Nhiệm vụ của nó là:
1. Cho phép người dùng chọn **Bài toán** và **Thuật toán** trực tiếp qua tham số terminal.
2. Tự động liên kết (cung cấp bộ Adapter) nếu bạn dùng Thuật toán liên tục (như GA, CS) để giải Bài toán rời rạc (Knapsack, Graph Coloring).
3. Thi hành việc tính toán, in ra tiến trình ở console và **vẽ biểu đồ hội tụ (Convergence Curve)** lưu vào `results/single_runs/`.

**Cú pháp sử dụng chung:**
```bash
python main.py --problem <Tên_Bài_Toán> --algo <Tên_Thuật_Toán> [các_tuỳ_chọn_khác]
```

**Ví dụ chạy thực tế:**
- Chạy thuật toán Kiến (ACO) cho bài toán Người chào hàng (TSP):
  ```bash
  python main.py --problem TSP --algo ACO --iterations 100
  ```
- Chạy thuật toán Di truyền (GA) trên hàm Ackley Liên tục với quần thể 50:
  ```bash
  python main.py --problem Ackley --algo GA --iterations 200 --pop_size 50
  ```
- Chạy Cuckoo Search (CS) để giải siêu bài toán Balo (Knapsack) (Dùng cơ chế liên tục giải rời rạc tự động):
  ```bash
  python main.py --problem Knapsack --algo CS --iterations 100
  ```

*Ghi chú: Nếu bạn không truyền đối số, file `main.py` sẽ hiển thị hướng dẫn (help) và liệt kê danh sách tương thích thuật toán.*

---

## 📊 2. Các Script Phân tích Nâng cao (`scripts/`)

Ngoài việc chạy đơn lẻ bằng `main.py`, dự án có một hệ thống Script chuyên nghiệp để xuất Báo cáo dữ liệu (Dashboard), đánh giá toàn diện tính chất của các thuật toán:

### A. So sánh Hiệu suất Tổng hợp (Benchmarking)
So sánh hàng loạt thuật toán cùng lúc và sinh ra các hình vẽ Radar, Heatmap, Bảng thống kê.
- **Chạy:** `python scripts/compare_all.py`
- *Kết quả ra ở:* `results/compare/`

### B. Phân tích Độ nhạy Tham số (Parameter Sensitivity)
Kiểm tra xem khi đổi 1 tham số cấu hình (ví dụ `w` của PSO, hay `F` của DE), hiệu quả thuật toán bị ảnh hưởng thế nào. Đã sinh sẵn biểu đồ Convergence tổng hợp.
- **Bài toán Rời rạc (TSP, Knapsack...):** `python scripts/param_sensitivity.py`
- **Bài toán Liên tục (Sphere, Ackley...):** `python scripts/param_sensitivity_continuous.py`
- *Kết quả ra ở:* `results/param_sensitivity/` và `results/param_sensitivity_continuous/`

### C. Đánh giá Độ Ổn định (Robustness)
Kiểm tra các thuật toán xem có bị phụ thuộc "may rủi" sinh số ngẫu nhiên không, bằng cách chạy lặp lại độc lập 30 lần và vẽ biểu đồ hình hộp (Boxplot / Violin).
- **Bài toán Rời rạc:** `python scripts/robustness_discrete.py`
- **Bài toán Liên tục:** `python scripts/robustness_continuous.py`
- *Kết quả ra ở:* `results/robustness_discrete/` và `results/robustness_continuous/`

### D. Đo Thời gian chạy thực tế (Execution Time)
Nhắm đo tốc độ thuật toán giải quyết bài toán qua Median (Trung vị) của 7 lần chạy, và sinh ra Bar chart / Heatmap siêu đẹp.
- **Bài toán Rời rạc:** `python scripts/exec_time_discrete.py`
- **Bài toán Liên tục:** `python scripts/exec_time_continuous.py`
- *Kết quả ra ở:* `results/exec_time_discrete/` và `results/exec_time_continuous/`

### E. Xuất Hình Động (GIFs Generator)
Bạn có thể xem được quá trình những quần thể (đàn kiến, lứa chim, đom đóm) hội tụ di chuyển về nghiệm đáy hàm như thế nào theo mỗi generation / iter qua dạng ảnh GIF.
- **Chạy:** `python scripts/generate_problem_gifs.py`
- *Kết quả ra ở:* `results/gifs/`

---

## 🛠 Yêu cầu Cài đặt (Requirements)

Đảm bảo bạn đã cài đặt Python 3.8+ và các thư viện hỗ trợ bằng pip:

```bash
pip install numpy matplotlib
```

*(Lưu ý: Bạn không cần cài các thư viện đặc biệt nào bên ngoài, phần lớn mã tự code nguyên bản thuần tính toán ma trận numpy).* 
