
import pickle
from pyvi import ViTokenizer
import numpy as np
import gensim

loaded_model = pickle.load(open('model_linear.sav', 'rb'))
tfidf_vect = pickle.load(open('tfidf_vect.pkl', 'rb'))

test_doc = '''Tiếp đón ĐT Việt Nam trên sân nhà ở bán kết lượt đi AFF Cup 2018, những sai lầm nơi hàng thủ đã khiến ĐT Philippines nhận thất bại cay đắng 1-2. Sau trận, một số cầu thủ liên tục đăng đàn thể hiện sự tiếc nuối với kết quả này, thậm chí tuyên bố đội nhà xứng đáng giành chiến thắng hơn. 

Philippines thua Việt Nam: Nội bộ lục đục, báo châu Á khó tin phép màu - 1

Patrick Reichelt chỉ trích thái độ thi đấu thiếu quyết tâm của các đồng đội

Tuy nhiên khá bất ngờ khi Patrick Reichelt - tác giả bàn gỡ 1-1 lại lên tiếng chỉ trích thái độ thi đấu của các đồng đội giữa thời điểm nhạy cảm. Phản ứng này khiến dư luận nghi ngờ về tình trạng lục đục nội bộ ở Philippines.

"Các cầu thủ chỉ chơi với 80-90% phong độ, điều đó không đủ giúp Philippines chiến thắng. Tôi không hề muốn dừng bước trong lần thứ 3 lọt vào bán kết AFF Cup nhưng nếu toàn đội thi đấu hết mình, tôi sẽ không cảm thấy hối tiếc dù thất bại. Philippines đã có sự chuẩn bị rất tốt, vấn đề nằm ở thái độ thi đấu", trích lời Reichelt trên Fox Sport Asia.

Trong khi đó, chuyên gia bóng đá Đông Nam Á nổi tiếng Gabriel Tan cũng phân tích khá chi tiết những điểm mạnh, điểm yếu của Philppines ở bài viết: "AFF Cup: Philippines vẫn còn cơ hội sống sót hay Việt Nam đã đặt một chân vào chung kết?".

"Philippines phần nào tái hiện được tinh thần và lối chơi từng giúp họ cầm hòa ĐKVĐ Thái Lan 1-1 ở vòng bảng. Thầy trò Sven-Goran Eriksson cũng gây ra nhiều khó khăn cho Việt Nam suốt 90 phút, thậm chí trở thành đội đầu tiên chọc thủng lưới Đặng Văn Lâm ở AFF Cup 2018".

Tuy nhiên, Gabriel Tan lại bỏ ngỏ khả năng thầy trò Eriksson lội ngược dòng khi hành quân tới Hà Nội vào ngày 6/12 tới và chỉ gợi lại kỉ niệm đẹp tại SVĐ Mỹ Đình 8 năm trước - thời điểm Philippines đánh bại Việt Nam 2-0:

"Philippines có thể lội ngược dòng? Không có gì đảm bảo cả. Việt Nam vẫn còn nhiều phương án chiến thuật, nhân sự cho khả năng tấn công biên, trong khi The Azkals chỉ còn 18 cầu thủ. Tới Hà Nội, HLV Eriksson chỉ biết hy vọng các học trò thể hiện tinh thần quyết tâm như trận hòa Thái Lan và tái hiện phép màu Hà Nội cách đây 8 năm".

Philippines thua Việt Nam: Nội bộ lục đục, báo châu Á khó tin phép màu - 2
"Phép màu Hà Nội 2010" là yếu tố để giới chuyên môn lẫn các cầu thủ Philippines bấu víu ở trận bán kết lượt về

Về màn trình diễn của ĐT Việt Nam, Gabriel Tan đánh giá rất cao HLV Park Hang Seo với những điều chỉnh chiến thuật, nhân sự cực kì táo bạo, nhạy bén: 

"Chiến thắng của Việt Nam ấn tượng hơn cả bởi HLV Park Hang Seo thậm chí chưa tung ra Văn Quyết, Xuân Trường, trong khi Công Phượng chỉ vào sân 10 phút cuối. Thay vào đó, Đức Huy và Hùng Dũng - những cầu thủ mới đá chính ở lượt trận cuối vòng bảng gặp Campuchia - được lựa chọn cho vị trí tiền vệ trung tâm.

Nhiều người cho rằng họ vào sân chỉ để giúp Xuân Trường dưỡng sức, tạo điều kiện cho Quang Hải trở về vị trí đá cánh sở trường, nhưng chiến lược gia người Hàn Quốc lại nghĩ khác. Ông không e ngại đặt niềm tin vào những cầu thủ trẻ. Mặt khác, hàng thủ với bộ ba hậu vệ, Đặng Văn Lâm và đôi cánh Trọng Hoàng - Văn Hậu tiếp tục cho thấy sự ăn ý đáng kinh ngạc".

'''


def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines


test_doc = preprocessing_doc(test_doc)
test_doc_tfidf = tfidf_vect.transform([test_doc])

loaded_model.predict(test_doc_tfidf)
