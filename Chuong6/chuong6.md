# <center> Welcome to Computer Vision</center>
 **<center>Trần Việt Anh - Hoàng Nguyên Phương</center>** 

## 6. Cấu hình môi trường để làm việc
Khi nói đến việc học một công nghệ mới (đặc biệt là học sâu), việc định cấu hình môi trường phát triển của bạn có xu hướng là một nửa trận chiến. Giữa các hệ điều hành khác nhau, các phiên bản phụ thuộc khác nhau và bản thân các thư viện thực tế, việc định cấu hình môi trường phát triển học sâu của riêng bạn có thể khá đau đầu. Tất cả những vấn đề này đều tăng thêm do tốc độ cập nhật và phát hành các thư viện học sâu - các tính năng mới thúc đẩy sự đổi mới, nhưng cũng phá vỡ các phiên bản trước đó. Đặc biệt, Bộ công cụ CUDA là một ví dụ tuyệt vời: trung bình có 2-3 bản phát hành CUDA mới mỗi năm. Với mỗi bản phát hành mới mang đến những tối ưu hóa, các tính năng mới và khả năng đào tạo mạng thần kinh nhanh hơn. Nhưng mỗi bản phát hành lại làm phức tạp thêm khả năng tương thích ngược. Chu kỳ phát hành nhanh này ngụ ý rằng học sâu không chỉ phụ thuộc vào cách bạn định cấu hình môi trường phát triển của mình mà còn khi bạn định cấu hình nó. Tùy thuộc vào khung thời gian, môi trường của bạn có thể lỗi thời! 

### 6.1 Libraries và Packages
Phần này trình bày chi tiết về ngôn ngữ lập trình cùng với các thư viện chính mà chúng tôi sẽ sử dụng để nghiên cứu học sâu về thị giác máy tính

#### 6.1.1 Keras

Để xây dựng và đào tạo mạng lưới học tập sâu của chúng tôi, chúng tôi sẽ chủ yếu sử dụng thư viện Keras. Keras hỗ trợ cả TensorFlow và Theano, giúp bạn dễ dàng xây dựng và đào tạo mạng một cách nhanh chóng. 

#### 6.1.2 Mxnet
Chúng tôi cũng sẽ sử dụng mxnet, một thư viện học sâu chuyên về học phân tán, đa máy. Khả năng đào tạo song song trên nhiều GPU

#### 6.1.4 OpenCV, scikit-image, scikit-learn, ...

Vì cuốn sách này tập trung vào việc áp dụng học sâu vào thị giác máy tính, chúng tôi cũng sẽ tận dụng một vài thư viện bổ sung.

### 6.2 Tổng kết

Khi nói đến cấu hình môi trường phát triển học sâu của bạn, bạn có một số tùy chọn. Nếu bạn muốn làm việc từ máy cục bộ của mình, điều đó hoàn toàn hợp lý, nhưng trước tiên bạn sẽ cần phải biên dịch và cài đặt một số phụ thuộc. Nếu bạn đang lên kế hoạch sử dụng GPU tương thích CUDA trên máy cục bộ của mình, bạn cũng cần thực hiện thêm một số bước cài đặt. Đa số sử dụng Linux nhưng chúng tôi sẽ cố thử trên colab



[Xem tiếp chương 6](Chuong7/chuong7.md)