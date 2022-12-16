# <center> Welcome to Computer Vision</center>
 **<center>Trần Việt Anh </center>** 

 ## 4. Phân loại hình ảnh cơ bản

 chúng tôi có thể thấy và mọi thứ , hình ảnh, nghệ thuật, nội dung một cách trực quan và có thể lưu trữ kiến thức để sử dụng sau này. uy nhiên, đối với máy tính, việc diễn giải nội dung của một hình ảnh kém hơn rất nhiều – tất cả những gì máy tính của chúng tôi nhìn thấy là một ma trận lớn các con số. Nó không biết gì về suy nghĩ, kiến thức hoặc ý nghĩa mà hình ảnh đang cố gắng truyền tải. Để hiểu nội dung của một hình ảnh, chúng tôi phải áp dụng phân loại hình ảnh, đó là nhiệm vụ sử dụng thuật toán thị giác máy tính và máy học để trích xuất ý nghĩa từ một hình ảnh. Hành động này có thể đơn giản như gán nhãn cho nội dung của hình ảnh hoặc nâng cao như diễn giải nội dung của hình ảnh và trả về một câu mà con người có thể đọc được. Phân loại hình ảnh là một lĩnh vực nghiên cứu rất rộng lớn, bao gồm nhiều kỹ thuật khác nhau – và với sự phổ biến của học sâu, Deep Learning đang tiếp tục phát triển. 
 Bây giờ là lúc để thúc đẩy làn sóng phân loại hình ảnh và học sâu – những người thực hiện thành công sẽ được khen thưởng hậu hĩnh. Phân loại hình ảnh và hiểu hình ảnh hiện đang (và sẽ tiếp tục là) lĩnh vực phụ phổ biến nhất của thị giác máy tính trong mười năm tới. Trong tương lai, chúng tôi sẽ thấy các công ty như Google, Microsoft, Baidu và những công ty khác nhanh chóng có được các công ty khởi nghiệp chuyên về hình ảnh thành công. chúng tôi sẽ thấy ngày càng nhiều ứng dụng dành cho người tiêu dùng trên điện thoại thông minh có thể hiểu và diễn giải nội dung của hình ảnh. Ngay cả các cuộc chiến tranh cũng có thể sẽ được tiến hành bằng cách sử dụng máy bay không người lái được hướng dẫn tự động bằng thuật toán thị giác máy tính.<b> Trong chương này, tôi sẽ cung cấp tổng quan cấp cao về phân loại hình ảnh là gì, cùng với nhiều thách thức mà thuật toán phân loại hình ảnh phải vượt qua. Chúng tôi cũng sẽ xem xét ba loại học tập khác nhau liên quan đến phân loại hình ảnh và học máy. </b>

 Cuối cùng, tôi sẽ kết thúc chương này bằng cách thảo luận về bốn bước đào tạo mạng học sâu để phân loại hình ảnh và cách so sánh quy trình bốn bước này với quy trình trích xuất tính năng thủ công truyền thống.

 ### 4.1 Phân loại hình ảnh là gì
 Phân loại hình ảnh, về bản chất, là nhiệm vụ gán nhãn cho một hình ảnh từ một tập hợp các loại được xác định trước. Thực tế, điều này có nghĩa là nhiệm vụ của chúng tôi là phân tích một hình ảnh đầu vào và trả về một nhãn phân loại hình ảnh. Nhãn luôn là từ một tập hợp các dữ liệu có thể được xác định trước. Ví dụ: giả sử rằng tập hợp các nhãn của dữ liệu của chúng tôi bao gồm: categories  = {cat, dog, panda} Sau đó, chúng tôi trình bày hình ảnh sau (Hình 4.1) cho hệ thống phân loại của mình:
 <center><img src="dog.png" width="300"/></center>
<center><font size="-1">Hình 4.1: : Mục tiêu của một hệ thống phân loại hình ảnh là lấy một hình ảnh đầu vào và gán nhãn dựa trên một tập hợp các danh mục được xác định trước. </font></center> 


Mục tiêu của chúng tôi ở đây là lấy hình ảnh đầu vào này và gán nhãn cho nó từ bộ danh mục của chúng tôi – trong trường hợp này là con chó. Hệ thống phân loại có thể ra nhiều nhãn cho 1 hình ảnh đó, ví dụ như : dog: 95%; cat: 4%; panda: 1%.

Chính xác hơn, với hình ảnh đầu vào của chúng tôi là W × H pixel với ba kênh tương ứng là Đỏ, Xanh lục và Xanh lam, mục tiêu của chúng tôi là lấy hình ảnh W × H × 3 = N pixel và tìm ra cách phân loại chính xác nội dung của bức hình

#### 4.1.1 Lưu ý về điểm dữ liệu
Khi thực hiện học máy và học sâu, chúng tôi có một bộ dữ liệu mà chúng tôi đang cố gắng trích xuất kiến ​​thức từ đó. Mỗi ví dụ/mục trong tập dữ liệu (dù là dữ liệu hình ảnh, dữ liệu văn bản, dữ liệu âm thanh, v.v.) đều là một điểm dữ liệu. Do đó, một bộ dữ liệu là một tập hợp các điểm dữ liệu (Hình 4.2)
 <center><img src="pointdata.png" width="300"/></center>
<center><font size="-1">Hình 4.2: :  bộ dữ liệu là một tập hợp các điểm dữ liệu </font></center> 

Mục tiêu của chúng tôi là áp dụng thuật toán học máy và học sâu để khám phá các mẫu cơ bản trong tập dữ liệu, cho phép chúng tôi phân loại chính xác các điểm dữ liệu mà thuật toán của chúng tôi chưa gặp phải. Bây giờ hãy dành thời gian để làm quen với thuật ngữ này:
1. Trong chủ đề phân loại hình ảnh, tập dữ liệu của chúng tôi là một tập hợp các hình ảnh.
2. Do đó, mỗi hình ảnh là một điểm dữ liệu.
Tôi sẽ sử dụng thuật ngữ hình ảnh và điểm dữ liệu thay thế cho nhau trong suốt phần còn lại của cuốn sách này, vì vậy hãy ghi nhớ điều này ngay bây giờ.

#### 4.1.2 Khoảng cách phương diện (The Semantic Gap)
 <center><img src="point_data.png"/></center>
<center><font size="-1">Hình 4.3: Trên cùng: Bộ não của chúng tôi có thể thấy rõ sự khác biệt giữa hình ảnh có con mèo và hình ảnh có con chó. Dưới cùng: Tuy nhiên, tất cả những gì máy tính "thấy" là một ma trận lớn các con số. Sự khác biệt giữa cách chúng tôi cảm nhận một hình ảnh và cách hình ảnh được thể hiện (một ma trận các số) được gọi là khoảng cách phương diện </font></center> 
chúng tôi có thể mô tả hình ảnh thông qua không gian, màu sắc, kết cấu. Vậy làm sao chúng tôi có thể mã hóa tất cả thông tin này theo cách mà máy tính có thể hiểu được ? Câu trả lời là áp dụng trích xuất tính năng để định lượng nội dung của hình ảnh. Trích xuất tính năng là quá trình lấy ảnh đầu vào, áp dụng thuật toán và thu được một vectơ đặc trưng (tức là danh sách các số) định lượng hình ảnh của chúng tôi.

Để thực hiện quy trình này, chúng tôi có thể cân nhắc áp dụng các tính năng được thiết kế thủ công như HOG, LBP hoặc các phương pháp tiếp cận “truyền thống” khác để định lượng hình ảnh. Một phương pháp khác, và phương pháp được sử dụng trong cuốn sách này, là áp dụng học sâu để tự động tìm hiểu một tập hợp các tính năng có thể được sử dụng để định lượng và cuối cùng là gắn nhãn cho chính nội dung của hình ảnh. Tuy nhiên, nó không đơn giản như vậy. . . bởi vì một khi chúng tôi bắt đầu kiểm tra hình ảnh trong thế giới thực, chúng tôi phải đối mặt với rất nhiều thách thức.
#### 4.1.3 Các thách thức
Nếu khoảng cách phương diện không đủ là một vấn đề, chúng tôi cũng phải xử lý các yếu tố biến thể trong cách một hình ảnh hoặc đối tượng xuất hiện. Hình 4.5 hiển thị hình ảnh trực quan của một số
các yếu tố biến đổi này.
 <center><img src="challenge.png"/></center>
<center><font size="-1">Hình 4.4: Khi phát triển một hệ thống phân loại hình ảnh, chúng ta cần nhận thức được cách một đối tượng có thể xuất hiện ở các góc nhìn, điều kiện ánh sáng, che khuất, tỷ lệ khác nhau, v.v.. </font></center> 
<ul>
    <li><b> viewpoint variation:</b> Có nhiều góc nhìn khác nhau, nhưng vẫn là thiết bị đó </li>
    <li><b> scale variation:</b> Có kích thước khác nhau nhưng vẫn là chiếc cốc đó </li>
    <li><b>deformation:</b> Có biến dạng nhiều cách khác nhau, nhưng vẫn là nhân vật hoạt hình đó </li>
    <li><b>occlusions:</b> Có bị che khuất, bị thoát khỏi tầm nhìn nhưng vẫn là con chó đó </li>
    <li><b>illumination:</b> Ở môi trường và điều kiện sáng khác nhau, nhưng vẫn là chiếc cốc đó </li>
    <li><b>background clutter:</b> Ở môi trường bị nhiều tiếng ồn và phức tạp khác nhau, thì vẫn có nhân vật đó trong bức ảnh </li>
    <li><b>intra-class variationr:</b> Chiếc ghế có sự đa dạng khác nhau, nhưng vẫn có thể phân loại chính xác các biến thể đó </li>
</ul>

Bạn bắt đầu cảm thấy hơi choáng ngợp với sự phức tạp của việc xây dựng bộ phân loại hình ảnh? Còn rất nhiều biến thể  khác vậy làm thế nào để chúng ta giải thích được số lượng biến thể đáng kinh ngạc như vậy trong các đối tượng.

Các hệ thống thị giác máy tính, phân loại hình ảnh và học sâu thành công được triển khai trong thế giới thực đưa ra các giả định và cân nhắc cẩn thận trước khi viết một dòng mã. Nếu bạn áp dụng cách tiếp cận quá rộng, chẳng hạn như “Tôi muốn phân loại và phát hiện từng đồ vật trong nhà bếp của mình”, (nơi có thể có hàng trăm đồ vật khả thi) thì hệ thống phân loại của bạn khó có thể hoạt động tốt trừ khi bạn có nhiều năm kinh nghiệm. xây dựng bộ phân loại hình ảnh – và thậm chí sau đó, không có gì đảm bảo cho sự thành công của dự án. Nhưng nếu bạn đóng khung vấn đề của mình và thu hẹp phạm vi, chẳng hạn như “Tôi chỉ muốn nhận ra bếp và tủ lạnh”, thì hệ thống của bạn có nhiều khả năng hoạt động và chính xác hơn, đặc biệt nếu đây là lần đầu tiên bạn làm việc với phân loại hình ảnh và học sâu. Điểm mấu chốt ở đây là luôn xem xét phạm vi của bộ phân loại hình ảnh của bạn. Mặc dù deep learning và Convolutional Neural Networks đã thể hiện sức mạnh đáng kể và khả năng phân loại trước nhiều thách thức khác nhau, nhưng bạn vẫn nên giữ phạm vi dự án của mình chặt chẽ và được xác định rõ ràng nhất có thể.

Hãy nhớ rằng ImageNet, bộ dữ liệu điểm chuẩn thực tế cho các thuật toán phân loại hình ảnh, bao gồm 1.000 đối tượng mà chúng ta gặp trong cuộc sống hàng ngày – và bộ dữ liệu này vẫn được các nhà nghiên cứu tích cực sử dụng để cố gắng thúc đẩy state-of-the-art của Deep Learning Forward. DeepLearning không phải là phép thuật. Thay vào đó, học sâu giống như một chiếc cưa trong nhà để xe của bạn – mạnh mẽ và hữu ích khi được sử dụng đúng cách, nhưng nguy hiểm nếu được sử dụng mà không có sự cân nhắc thích hợp. Trong suốt phần còn lại của cuốn sách này, tôi sẽ hướng dẫn bạn về hành trình học sâu và giúp chỉ ra khi nào bạn nên tiếp cận với những công cụ mạnh mẽ này và khi nào bạn nên tham khảo một cách tiếp cận đơn giản hơn (hoặc đề cập nếu một vấn đề không hợp lý đối với hình ảnh). phân loại để giải).

### 4.2 Các loại học 