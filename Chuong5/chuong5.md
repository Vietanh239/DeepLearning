# <center> Welcome to Computer Vision</center>
 **<center>Trần Việt Anh - Hoàng Nguyên Phương</center>** 

## 5. Bộ dữ liệu cho phân loại ảnh
Tại thời điểm này, tác giả đã quen với các nguyên tắc cơ bản của quy trình phân loại hình ảnh - nhưng trước khi chúng tôi đi sâu vào bất kỳ dòng code nào để xem thực sự cách lấy tập dữ liệu và xây dựng bộ phân loại hình ảnh, trước tiên hãy xem lại các tập dữ liệu mà bạn sẽ thấy bên trong Deep Learning cho Thị giác máy tính với Python

Một số bộ dữ liệu này về cơ bản đã được “giải quyết”, cho phép chúng tôi có được các bộ phân loại có độ chính xác cực cao (độ chính xác> 95%) mà không tốn nhiều công sức. Các bộ dữ liệu khác đại diện cho các hạng mục về thị giác máy tính và các vấn đề học sâu vẫn là những chủ đề nghiên cứu mở ngày nay và còn lâu mới giải quyết được.

### 5.1 Mnist

<center><img src="https://storage.googleapis.com/tfds-data/visualization/fig/mnist-3.0.1.png" width="300"/></center>
<center><font size="-1">Hình 5.1: Bộ dữ liệu Mnist</font></center>

MNIST (“NIST” là viết tắt của National Institute of Standards and Technology trong khi “M” là viết tắt của “modified” vì dữ liệu đã được xử lý trước để giảm bớt gánh nặng cho việc xử lý thị giác máy tính và chỉ tập trung vào nhiệm vụ nhận dạng chữ số) tập dữ liệu là một trong những bộ dữ liệu được nghiên cứu nhiều nhất về thị giác máy tính và tài liệu học máy.

Bản thân MNIST bao gồm 60.000 hình ảnh đào tạo và 10.000 hình ảnh thử nghiệm. Mỗi vectơ đặc trưng là 784-chiều, tương ứng với cường độ pixel 28 × 28 thang độ xám của hình ảnh. Các cường độ pixel thang độ xám này là các số nguyên không dấu, rơi vào phạm vi [0,255]. Tất cả các chữ số được đặt trên nền đen với nền trước là màu trắng và các sắc thái của màu xám. Với các cường độ pixel thô này, mục tiêu của chúng tôi là đào tạo mạng nơ-ron để phân loại chính xác các chữ số.

### 5.2 Động vật: chó,mèo,gấu panda


