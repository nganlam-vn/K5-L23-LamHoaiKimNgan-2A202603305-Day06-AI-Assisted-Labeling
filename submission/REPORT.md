# Báo cáo Day 08 — 2A202603305 - Lâm Hoài Kim Ngân

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | 0.3 (mặc định) | Theo kết quả `threshold_sweep` trên 4 ảnh gold, không có ngưỡng nào <= 0.5 đạt recall >= 90% cho person, motorbike, car (tối đa person 72%, motorbike 64%, car 88%), còn bicycle, bus, truck có < 5 mẫu trong gold nên giữ mặc định 0.3. Thử nghiệm pilot (r01 conf 0.25 vs r02 conf 0.5): ngưỡng 0.5 giảm nhiễu nền nhưng làm mất >35% vật nhỏ và người đi bộ ở xa, do đó chọn 0.3 để cân bằng precision và recall. Dẫn chứng: `r01`, `r02`, `r05`. |
| three_wheeler | YOLOE-26s, prompt: three_wheeler | 0.15 | YOLO26 là detector cố định 80 lớp COCO nên không có nhãn tuk-tuk, thường bỏ sót hoặc nhầm thành car/truck. YOLOE-26s open-vocabulary nhận diện tốt phương tiện ba bánh (`r03`). Quét ngưỡng trên gold (`r06`) cho thấy tại ngưỡng 0.15, YOLOE đạt recall 100% (vượt mục tiêu >= 90%) và precision 78% trên 7 vật gold (`submission/qc/threshold_sweep.md`). Dẫn chứng: `r03`, `r06`. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn **SAM 2.1** (trọng số `sam2.1_b.pt` - Base).
  - *Chất lượng mask*: Cho đường viền polygon sắc nét, bám sát biên dạng người và phương tiện phức tạp, vượt trội so với MobileSAM (SAM 1).
  - *Tốc độ và RAM*: Bản Base dung lượng ~400MB, chạy ổn định trên CPU máy cá nhân (~7.58s/ảnh trên pilot `r04`) mà không gây tràn bộ nhớ (OOM) như SAM 2.1 Large hay SAM 3.
  - *Quyền truy cập*: Sử dụng trọng số mã nguồn mở tải trực tiếp, không yêu cầu Hugging Face token hay cổng kiểm duyệt bản quyền như SAM 3.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào?
  - *Lỗi không sửa được bằng đổi ngưỡng*: Lỗi sai lệch ngữ nghĩa do giới hạn nhãn (taxonomy mismatch): YOLO26 không có lớp `three_wheeler` nên tuk-tuk luôn bị nhận nhầm thành `car` hoặc `truck` dù tăng hay giảm `conf` (ở conf 0.5 trong `r02` tuk-tuk rõ vẫn bị gán `car`, còn conf 0.05 trong `r05` chỉ làm tăng box rác). Ngoài ra, ngưỡng không sửa được việc box YOLOE bị hở viền nền (`loose`) hoặc SAM gộp dính người lái vào thân xe máy.
  - *Cách sửa*: Kết hợp mô hình open-vocabulary YOLOE cho lớp `three_wheeler`; khi chạy `merge_prelabels.py`, đặt file YOLOE lên trước YOLO26 kèm ngưỡng IoU `--iou 0.6` để box `three_wheeler` đè và loại bỏ box `car`/`truck` nhầm của YOLO26 trên cùng một xe (đã loại bỏ 11 box `car` và 6 box `truck` bị trùng). Ở bước review trên CVAT, áp dụng quy trình kiểm tra 4 bước để nắn khít box lỏng và tách độc lập người lái.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - *Điểm cao mà sai*:
    - Trên `img_013.jpg`: YOLO26 gán chiếc xe tuk-tuk lớn ở tiền cảnh thành `car` với điểm tin cậy cao `conf = 0.415` (cao hơn ngưỡng 0.3).
    - Trên `img_002.jpg`: YOLOE gán nhầm chiếc xe tải ở nền sau (`[863.0, 426.3, 161.0, 129.7]`) thành `three_wheeler` với điểm `conf = 0.366`.
  - *Điểm thấp mà đúng*:
    - Trên `img_002.jpg`: Người ngồi bên trong tuk-tuk (`[417.6, 458.7, 45.7, 73.8]`, `conf = 0.357`) và người đi bộ nhỏ ở xa (`[149.1, 467.4, 24.8, 28.3]`, `conf = 0.273`) có điểm số thấp (sẽ bị mất nếu đặt ngưỡng 0.5) nhưng là đối tượng thật.
    - Trên `img_013.jpg`: Xe tuk-tuk nhỏ ở hậu cảnh xa chỉ đạt điểm `three_wheeler = 0.183` từ YOLOE (nằm dưới ngưỡng mặc định 0.3, nhưng nhờ hạ ngưỡng lớp xuống 0.15 nên được giữ lại) và là đối tượng đúng.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan:
  - Đây là hiện tượng *in-sample bias* (overfitting trên tập chọn ngưỡng): vừa tìm ngưỡng tối ưu trên 4 ảnh gold vừa đánh giá lại precision/recall trên chính 4 ảnh đó.
  - Kích thước mẫu 4 ảnh quá nhỏ (các lớp `bus`, `truck`, `bicycle` có 0 đến 1 đối tượng, `three_wheeler` chỉ có 7 đối tượng) nên không mang tính đại diện thống kê cao và độ tin cậy thấp.
  - Trên 24 ảnh batch thực tế, bối cảnh phức tạp và độ biến thiên mật độ giao thông cao hơn nhiều, do đó các chỉ số precision và recall thực tế của máy sẽ sụt giảm so với số đo lạc quan trên tập gold.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
