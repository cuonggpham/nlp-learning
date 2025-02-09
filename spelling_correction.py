#pip install pyspellchecker 
 

from spellchecker import SpellChecker
spell = SpellChecker()

def correct_spellings(text):
    corrected_text = []
    # Tách từ từ chuỗi đầu vào
    words = text.split()
    # Lấy danh sách từ sai chính tả
    misspelled_text = spell.unknown(words)
    print("Misspelled text: ", misspelled_text)
    for word in words:
        # Kiểm tra nếu từ sai chính tả thì sửa
        if word in misspelled_text:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
            
    return " ".join(corrected_text) 


text = input("Enter Text: ")
result = correct_spellings(text)
print("Corrected Text:", result)
