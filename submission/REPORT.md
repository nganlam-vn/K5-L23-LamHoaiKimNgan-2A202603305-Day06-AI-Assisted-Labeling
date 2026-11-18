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
| accepted | 485 | 485/523 = 92.7% | 523 box máy |
| edited | 24 | 24/523 = 4.6% | 523 box máy |
| deleted | 14 | 14/523 = 2.7% | 523 box máy |
| added | 8 | 8/517 = 1.5% | 517 đối tượng sau duyệt |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  - Lớp máy tệ nhất là `three_wheeler` và `motorbike` ở cụm xe dày đặc. Với `three_wheeler`, do YOLO26 không nhận được tuk-tuk nên phải dựa hoàn toàn vào YOLOE-26s, tuy nhiên YOLOE thường sinh box lỏng hoặc vẽ 2 box đè lên cùng 1 xe tuk-tuk (như trên `img_013.jpg` sinh 2 box đè nhau). Với `motorbike`, máy thường nhận nhầm các phần bóng đổ, giỏ hàng hoặc bỏ sót nhiều xe máy nhỏ ở hậu cảnh (ví dụ `img_018.jpg` bị sót xe máy nhỏ và sinh box rác).
- So với gold: precision / recall của **máy**: Precision đạt 50.0% (33/66, KTC 95%: 38–62%), Recall đạt 73.3% (33/45, KTC 95%: 59–84%) (mẫu số: 45 đối tượng gold trên 4 ảnh gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 11/12 = 91.7% (KTC 95%: 65–99%); loại nào lọt: 1 bẫy `loose` trên `img_018.jpg` (box xe máy bị lỏng); vì sao (bạn đã bỏ bước nào của checklist)? Do ảnh `img_018.jpg` có mật độ xe máy quá dày đặc (45 đối tượng trong một khung hình) dẫn đến mỏi mắt và phân tán chú ý, người duyệt đã bỏ qua bước 4 của checklist (rà soát kỹ 4 cạnh viền của từng box xe tiền cảnh).
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Nếu accepted cao (>95%) mà bắt bẫy thấp (<80%), điều đó phản ánh người duyệt bị thiên kiến tự động hóa (automation bias) nặng: lướt qua nhanh và chấp nhận máy vẽ sẵn mà không thẩm định bằng mắt. Ngược lại, việc bắt được 11/12 bẫy (91.7%) và duy trì tỷ lệ edited/deleted ~7.3% chứng minh người duyệt đã áp dụng checklist 4 bước, chủ động quét mắt phát hiện box thừa, sửa sai lớp và thêm vật sót thay vì thụ động tin vào máy.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 5.5 phút/ảnh (22 phút / 4 ảnh gold) vs duyệt 2.17 phút/ảnh (52 phút / 24 ảnh batch). Thiên lệch: khi gán 4 ảnh gold trước, người gán đã làm quen bối cảnh đường phố châu Á, phân biệt rõ các loại xe và ghi nhớ quy tắc guideline nên khi chuyển sang duyệt 24 ảnh batch đã có lợi thế quen tay hơn. Dù vậy, mức giảm thời gian từ 5.5 phút xuống 2.17 phút/ảnh (tiết kiệm ~60% thời gian) là rất đáng kể.
- Kết luận: Có, rất đáng dùng cho các lớp phổ biến như `person`, `car`, `motorbike` (tiền cảnh rõ) vì máy tạo khung bao ban đầu rất tốt, giảm gánh nặng vẽ mới; tuy nhiên với lớp đặc thù như `three_wheeler` hoặc các vùng xe máy chen chúc phức tạp, con người vẫn bắt buộc phải can thiệp kỹ để loại box trùng lặp và gom vùng ignore.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Luôn ẩn toàn bộ nhãn máy (bấm icon con mắt trong CVAT) trong 15 giây đầu của mỗi ảnh để quét mắt độc lập và đếm nhẩm số lượng người/xe trước khi bật đối chiếu.
  2. Rà soát bắt buộc 4 cạnh viền đối với ít nhất 100% các box ở tiền cảnh (chiều cao > 40px) để chống bẫy loose.
  3. Định kỳ nghỉ giải lao 3-5 phút sau mỗi 8-10 ảnh để tránh hiện tượng mỏi mắt và suy giảm nhận thức dẫn tới lướt ẩu.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013.jpg | SAM 2.1 Base (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không (đã tách người lái ra ngoài) | Cần 1 click âm (`720,300`) vào vị trí người lái để loại bỏ người ra khỏi mask thân xe tuk-tuk | tuktuk_img013_sam2.jpg |
| img_013.jpg | MobileSAM (`mobile_sam.pt`) | 3 (2 pos, 1 neg) | Không | Mask bị tách thành 2 polygon rời rạc (245.5k px vs 317.2k px của SAM 2.1), viền bánh xe bị răng cưa | tuktuk_img013_sam1.jpg |
| img_004.jpg | SAM 2.1 Base (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không (loại người đạp xích lô) | Cắt viền khung xe và mui xích lô rất sắc nét, click âm loại sạch người đạp phía sau | tuktuk_img004_sam2.jpg |
| img_017.jpg | SAM 2.1 Base (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không (chỉ lấy thân tuk-tuk) | Click âm tại `645,355` chặn mask không lấn sang người lái và cột cờ trên xe | tuktuk_img017_sam2.jpg |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? Quy tắc "Người là vật riêng; xe là vật riêng" và "Bóng đổ không thuộc vật" giúp định hướng đặt click âm chính xác để tách người lái và bóng gầm xe. Tuy nhiên guideline còn thiếu quy định cụ thể về việc xử lý các khoảng rỗng/khoảng hở nhìn xuyên qua khung kính xe ba bánh hay nan hoa bánh xe (SAM thường tự động lấp đầy phần khoảng hở này).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: Interactor SAM online trong CVAT yêu cầu backend serverless Nuclio với GPU đủ lớn, trên môi trường Docker cá nhân chỉ có CPU thường gặp lỗi timeout; chạy offline bằng script `sam2_points.py` ổn định hơn, lưu được tọa độ click tái lập (reproducible).

## 6. Calibration (Part 6)

- Số quyết định: 4 (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): CAL-02 (giải quyết triệt để lỗi máy YOLO26 nhầm tuk-tuk thành car trên `img_013.jpg` bằng pipeline ưu tiên YOLOE) và CAL-01 (chốt phân loại xe đạp điện/xe máy điện trên `img_003.jpg`).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: Bổ sung quy tắc phân loại xe điện (mục 4.1.8: có bàn đạp là `bicycle`, không bàn đạp là `motorbike`), chốt quy tắc hàng chở nhô ra quá 20% chiều dài thân xe (mục 4.1.5), và bắt buộc gán `three_wheeler` cho mọi xe ba bánh (mục 4.1.9).
- Nguyên nhân gốc: Do giới hạn danh mục mô hình (YOLO26 thiếu lớp `three_wheeler` trong COCO 80 nhãn) kết hợp với các tình huống biên phức tạp của giao thông đường phố châu Á mà guideline v1.0 chưa bao quát hết.

## 7. QC và quyết định (Part 6)

- Mẫu: 8 ảnh (3 ảnh `can_review` + 5 ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt nên dễ lặp lại cùng thiên kiến nhận thức cá nhân).
- Lỗi theo **đối tượng**: 2/192 = 1.0% · theo **ảnh**: 2/8 = 25.0% (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại:
  - `thua`: 1 lỗi (trên `img_013.jpg`, box #131 và #132 trùng lặp 2 khung three_wheeler đè lên nhau do YOLOE).
  - `sai_vi_tri_kich_thuoc`: 1 lỗi (trên `img_018.jpg`, box #204 xe máy bị lỏng viền do bẫy loose lọt).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): 2 ảnh lỗi trong 8 ảnh mẫu, cận trên khoảng tin cậy 95% (Wilson) của tỷ lệ ảnh lỗi thật lên tới **59%**. Ngưỡng 2% không thể được chứng minh tuyệt đối bằng mẫu nhỏ 8 ảnh.
- Script kiểm được / không kiểm được: Script kiểm tra được tính hợp lệ của cấu trúc file COCO, trùng lặp ID, giới hạn box ngoài ảnh và ngưỡng chiều cao tối thiểu; tuy nhiên script không thể kiểm tra được mức độ ôm sát của box với vật thể thực tế, sự phù hợp ngữ nghĩa hay việc cả người lẫn máy cùng bỏ sót vật thể trong bóng tối.

**Quyết định: Rework**, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release): Guideline đã được cập nhật rõ ràng ở v1.1 và dữ liệu ảnh gốc không lỗi (vượt qua tầng Hold); tuy nhiên tại tầng Rework, tỷ lệ ảnh có lỗi trong mẫu QC đạt 25.0% (2/8 ảnh), vượt quá ngưỡng 2.0% của dự án.
Nếu Rework / Hold, việc cần làm tiếp: Mở lại các ảnh bị lỗi (`img_013.jpg` để xoá box thừa, `img_018.jpg` để kéo khít box lỏng), rà soát thêm các ảnh giao thông đông đúc, cập nhật bản sửa vào `qc_fixed.json` và chạy lại `build_release.py`.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Dùng YOLO26m, YOLOE-26s và SAM 2.1 Base theo kịch bản pipeline của bài lab; dùng trợ lý AI hỗ trợ tự động hóa các thao tác xử lý dữ liệu và kiểm tra cấu trúc file.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không, 4 ảnh gold được thực hiện gán tay 100% độc lập hoàn toàn trước khi tiếp cận bất kỳ kết quả pre-label nào.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11, CPU Intel Core i7 (chạy inference CPU cục bộ), CVAT Docker v2.81.0; vướng mắc: cần xử lý encoding UTF-8 và bao ngoặc kép cho tham số lệnh trên PowerShell.
- Dùng cache `data/cache/` thay vì tự chạy? Không, tự chạy 100% tất cả các mô hình trên máy cục bộ.
- Lỗi / bug script đã gặp (nếu có): Lỗi phân tách tham số có dấu phẩy trên PowerShell khi không bọc ngoặc kép, đã giải quyết bằng cách luôn bọc dấu ngoặc kép cho các chuỗi tham số `--thr`, `--classes`, `--pos`.
- Một điều bạn làm khác lần sau: Sẽ chủ động nghỉ ngơi sau mỗi 5-7 ảnh phức tạp để duy trì độ tập trung cao nhất khi kiểm tra các góc viền nhỏ của phương tiện giao thông.
