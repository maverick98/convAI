Execute This.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Modify SentenceTransformer.py

If you want to keep huggingface_hub==0.14.1, edit the sentence_transformers source code:

    Open:

C:\software\anaconda3\Lib\site-packages\sentence_transformers\SentenceTransformer.py

Find:

from huggingface_hub import cached_download

Replace it with:

from huggingface_hub import hf_hub_download as cached_download

Then
sh install.sh
# convAI
