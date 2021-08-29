# How to train a new language model from scratch using Transformers and Tonkenizers

Trong vài tháng qua, Hugging Face đã thực hiện việc cải tiến đối với thư viện ``transformers`` và ``tokenizers`` của họ, với mục đích để giúp cho việc train một Langugage model mới từ đầu (from scratch) dễ dàng hơn.

Trong phần này, chúng ta sẽ demo cách để train một `small` model (tầm khoảng 84M parameters = 6 layers, 768 hidden size, 12 attention head) - giống với số layers và heads của `DistillBERT` trên bộ dữ liệu `Esperanto`. Sau đó, chúng ta sẽ thử fine-tune model đó trên downstream task POS (part-of-speech tagging).

