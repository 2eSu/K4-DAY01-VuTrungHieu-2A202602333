# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
        {
                "class_id": 468,
                "class_name": "cab",
                "rank": 1,
                "score": 0.510915,
                "taxonomy_name": "ImageNet-1K"
        }
- Record này mô tả toàn ảnh như thế nào?
        Trong bài toán image classification, model đưa ra một class đại diện cho toàn bộ ảnh, thay vì xác định vị trí cụ thể của từng object.
        Với ảnh 'traffic', prediction hạng 1 là 'cab'. Vì vậy, model nhận định rằng nội dung của toàn ảnh phù hợp nhất với class 'cab'. Classification không cung cấp bounding box hoặc vị trí của object. 
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
        Class list được xác định bởi taxonomy/dataset mà checkpoint được huấn luyện theo. Trong output này, 'taxonomy_name' là 'ImageNet-1K', nên các class mà 'yolo11n-cls.pt' dự đoán thuộc vocabulary của taxonomy này. Do đó, model không tự tạo ra danh sách class khi inference; checkpoint đã được huấn luyện để dự đoán trong một tập class đã xác định.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
        Cần giữ cả ba để prediction có thể được xác định và truy nguyên rõ ràng:
                'class_id': định danh class trong hệ thống.
                'class_name': tên class để con người dễ đọc và hiểu.
                'taxonomy_name': cho biết class ID và tên class đang thuộc hệ thống taxonomy nào.
        Nếu chỉ lưu 'class_id' = 468 thì chưa đủ ngữ cảnh để biết ID đó có ý nghĩa gì. Việc giữ cả ID, tên và taxonomy giúp giảm nhầm lẫn và thuận tiện cho việc kiểm tra, đối chiếu prediction.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
        Guideline cần quy định cách chọn class đại diện cho toàn ảnh khi có nhiều chủ thể, để các annotator đưa ra nhãn nhất quán.
        Cụ thể, guideline nên làm rõ:
                Chủ thể nào được xem là chủ thể chính.
                Nếu có nhiều chủ thể quan trọng ngang nhau thì chọn nhãn như thế nào.
                Những trường hợp ảnh mơ hồ hoặc không thể xác định class thì xử lý ra sao.
                Khi annotator không thể quyết định thì trường hợp nào cần review/escalation.
- Vì sao model score không phải ground truth?
        'score' là kết quả do model dự đoán, thể hiện mức độ model đánh giá prediction đó phù hợp với ảnh. Nó không phải nhãn đúng được xác nhận.
        Ví dụ với 'traffic':
                cab              0.510915
                minibus          0.164284
                police_van       0.085848
        cab đứng hạng 1 vì có score cao nhất, nhưng điều đó không chứng minh cab là ground truth. Model hoàn toàn có thể dự đoán sai.
        Ground truth phải đến từ annotation/guideline và quá trình kiểm tra chất lượng, trong khi score là output của model.
                Prediction = model nghĩ gì.
                Ground truth = nhãn được xác định là đúng theo quy trình annotation/QC.
        Đây là lý do không được dùng score để thay thế ground truth.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
        {
                "class_name": "person",
                "score": 0.912625,
                "bbox_xyxy": [
                        385.33,
                        69.24,
                        498.92,
                        348.92
                ],
                "bbox_width": 113.58,
                "bbox_height": 279.68
        }
- Diễn giải vị trí box bằng lời: 
        Bounding box của person nằm ở khu vực bên phải của ảnh.
        Box bắt đầu khoảng: x = 385.33, y = 69.24
        và kết thúc khoảng: x = 498.92, y = 348.92.
        Vì vậy, box bao quanh người đang đứng ở phía bên phải khu vực bếp, từ phần đầu/thân phía trên xuống gần chân.
- So sánh số prediction ở hai threshold:
        Threshold: 0.2          Predictions: 17
        Threshold: 0.35         Predictions: 11
        Threshold: 0.6          Predictions: 6
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
        Threshold thấp (0.20):
                Giữ lại nhiều prediction hơn: 17 objects.
                Có độ bao phủ tốt hơn, vì ít khả năng bỏ sót những object có score thấp.
                Tuy nhiên, có thể xuất hiện nhiều prediction sai hoặc không chắc chắn.
                Reviewer phải kiểm tra nhiều object hơn.
        Threshold trung bình (0.35):
                Còn 11 objects.
                Là sự cân bằng giữa độ bao phủ và số lượng prediction cần kiểm tra.
                Trong output hiện tại, threshold 0.35 được sử dụng để lưu prediction.
        Threshold cao (0.60):
                Chỉ còn 6 objects.
                Giảm đáng kể khối lượng reviewer phải kiểm tra.
                Tuy nhiên, có nguy cơ bỏ sót những object thật nhưng model có score thấp.
- Đề xuất một quy tắc box chặt:
        Bounding box phải bao phủ toàn bộ phần object nhìn thấy trong ảnh, bám sát biên của object, hạn chế tối đa background và không cắt mất phần object đang nhìn thấy.
        Cụ thể:
                Box phải chứa toàn bộ phần object có thể quan sát được.
                Không lấy quá nhiều background xung quanh.
                Không để box cắt mất phần object nhìn thấy.
                Nếu có nhiều object cùng lớp, mỗi object cần có một bounding box riêng.
        Ví dụ trong ảnh, mỗi bowl được model phát hiện sẽ có một box riêng thay vì gộp tất cả các bowl thành một box.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
        Guideline cần quy định rõ phạm vi của bounding box đối với object bị che khuất hoặc bị cắt mép.
        Ví dụ cần xác định:
                Nếu object chỉ bị che một phần: box có bao quanh phần nhìn thấy hay cố gắng suy ra toàn bộ object?
                Nếu object bị cắt bởi mép ảnh: box có được phép chạm sát mép ảnh không?
                Nếu không thể xác định rõ object thuộc về đâu hoặc biên của object không rõ, reviewer có cần escalate để người có thẩm quyền quyết định hay không?
        Một quy tắc an toàn là: Chỉ annotate phần object có thể xác định rõ từ ảnh; nếu trường hợp bị che khuất/cắt mép quá mơ hồ và guideline không quy định rõ, cần chuyển cho reviewer/escalation thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
        `instance_id` = kitchen-001
        `class_name`  = person
        `score`       = 0.899318
        Box của instance này là: [385.45, 66.44, 498.02, 348.58]. và polygon được biểu diễn bởi `polygon_point_count` = 348 điểm (x, y). 

- Polygon bổ sung chi tiết gì so với box?
        Polygon bổ sung thông tin về hình dạng và đường biên chi tiết của object so với bounding box. Bounding box chỉ xác định một vùng hình chữ nhật chứa object, trong khi polygon có thể mô tả sát hình dạng thực tế của object.
- `instance_id` dùng để làm gì và không phải loại ID nào?
        `instance_id` dùng để phân biệt từng object cụ thể trong cùng một ảnh. Ví dụ nhiều object đều có class_name = bowl nhưng có các `instance_id` khác nhau. `instance_id` không phải class_id, image ID hay confidence score.
- Đề xuất một quy tắc biên mask: Biên mask nên bám sát đường biên của phần object nhìn thấy, bao phủ đầy đủ object và hạn chế background hoặc phần của object khác. Với hình dạng phức tạp cần đủ số điểm để mô tả biên.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
        Với vùng bị mờ, tiếp xúc hoặc che khuất, guideline cần quy định rõ cách xác định ranh giới giữa các instance. Nếu không thể xác định chắc chắn, cần escalation cho reviewer thay vì tự suy đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
|Phân loại ảnh| Một `class_id`/`class_name` cho toàn ảnh theo taxonomy đã quy định | Ảnh có nhiều chủ thể hoặc nội dung không rõ lớp chính; có thể nhầm giữa các lớp gần nhau| Gán đúng một lớp theo guideline; nếu không xác định được thì đánh dấu/escalate thay vì tự đoán| Kiểm tra class có đúng guideline/taxonomy không và xử lý các trường hợp mơ hồ|
|Phát hiện vật thể| Mỗi object gồm `class_name` và `bbox` (`xyxy`)| Box quá rộng/hẹp, chứa nhiều background, cắt mất object; object bị che khuất hoặc nằm sát mép ảnh    | Tạo một box cho từng object, bám sát phần object nhìn thấy; xử lý theo guideline khi object bị che/cắt | Kiểm tra class, vị trí và độ chặt của box; kiểm tra các trường hợp che khuất/cắt mép |
|Instance segmentation| Mỗi instance gồm `instance_id`, `class_name` và `polygon`| Polygon lệch biên, chứa background, lấn sang object khác; biên bị mờ, object tiếp xúc hoặc che khuất | Vẽ polygon theo biên của từng instance và giữ các instance tách biệt| Kiểm tra class, instance và độ chính xác của biên mask; xem xét các vùng mơ hồ|

        Quy trình kiểm soát chất lượng bắt đầu từ ảnh thô, sau đó áp dụng guideline để tạo ground truth trước khi sử dụng dữ liệu cho huấn luyện và tạo prediction. Kết quả prediction được đưa qua bước QC để phát hiện lỗi và thực hiện rework khi cần. Với classification, QC tập trung vào tính chính xác của class cho toàn ảnh. Với object detection, QC kiểm tra class và bounding box của từng object. Với instance segmentation, QC kiểm tra class, instance và độ chính xác của polygon. Các trường hợp mơ hồ như object bị che khuất, cắt mép hoặc có ranh giới không rõ cần được xử lý theo guideline hoặc chuyển reviewer để đảm bảo tính nhất quán.

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng dữ liệu cho đúng mục đích của bài thực hành, không chia sẻ ảnh, dữ liệu hoặc kết quả ra bên ngoài phạm vi được phép và không đưa thông tin cá nhân vào báo cáo/output.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Tôi sẽ dừng xử lý dữ liệu đó và báo cho người phụ trách khóa học/lab hoặc reviewer được chỉ định để xác định cách xử lý. Tôi không tự ý tiếp tục annotate, sao chép hoặc chia sẻ dữ liệu.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
