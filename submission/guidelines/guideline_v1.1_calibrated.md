# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603305 - Lâm Hoài Kim Ngân

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện: bicycle hay motorbike? | Xe 2 bánh chạy điện có bàn đạp cơ/trợ lực tính là `bicycle`; xe máy điện không có bàn đạp (kiểu dáng xe tay ga/xe máy) tính là `motorbike`. | img_003.jpg | CAL-01 |
| 3 | Hàng hoá chở cồng kềnh nhô ra ngoài thân xe: box có gồm hàng không? | Hàng hoá nhô ra $\le 20\%$ chiều dài xe thì box xe bao trọn cả hàng; nếu nhô ra $> 20\%$ chiều dài xe thì box chỉ ôm thân xe, không tính phần hàng nhô dài phía sau. | img_024.jpg | CAL-03 |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_013.jpg | Xe tuk-tuk chở khách tiền cảnh bị YOLO26 nhận nhầm | YOLO26m gán nhầm thành `car` (conf = 0.415) do COCO không có nhãn xe ba bánh | Đổi lớp thành `three_wheeler`; cấu hình pipeline ưu tiên YOLOE đè lên YOLO26 | Mục 4.1.9: Mọi phương tiện 3 bánh chở người hoặc hàng (tuk-tuk, ba gác, xích lô) bắt buộc gán `three_wheeler` |
| img_018.jpg | Bẫy loose trên box xe máy bị lọt do cảnh đông đúc | Script gài bẫy loose làm box xe máy bị co nhỏ/lệch viền | Gom cụm xe nhỏ $< 20\text{ px}$ ở xa thành `ignore_region`, rà khít 4 cạnh các xe tiền cảnh | Mục 4.1.6: Thực hiện nghiêm ngặt checklist bước 4: kéo sát 4 cạnh viền box phương tiện tiền cảnh |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | Khung quy chuẩn ban đầu |
| v1.1 | Chốt phân loại xe đạp điện/xe máy điện; quy định hàng chở nhô ra quá 20%; chuẩn hóa gán nhãn `three_wheeler` cho tuk-tuk và siết chặt kiểm tra viền box xe máy | Khắc phục câu hỏi mở và lỗi máy (CAL-01, CAL-02, CAL-03, CAL-04) |
