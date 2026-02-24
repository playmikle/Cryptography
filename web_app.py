import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import Counter
import time
import numpy as np

def caesar_encrypt(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            shift_base = 65 if char.isupper() else 97
            result += chr((ord(char) - shift_base + shift) % 26 + shift_base)
        else:
            result += char
    return result

def caesar_decrypt(text, shift):
    return caesar_encrypt(text, -shift)

def caesar_bruteforce(text):
    results = ""
    for shift in range(1, 26):
        decrypted = caesar_decrypt(text, shift)
        results += f"Сдвиг {shift:2d}: {decrypted[:60]}{'...' if len(decrypted) > 60 else ''}\n"
    return results

def frequency_analysis(text):
    letters = [char.lower() for char in text if char.isalpha()]
    freq = Counter(letters)
    total = len(letters)
    if total == 0:
        return "Нет букв для анализа"
    freq_percent = {k: round(v/total*100, 1) for k, v in freq.items()}
    sorted_freq = dict(sorted(freq_percent.items(), key=lambda x: x[1], reverse=True)[:10])
    result = "Частотный анализ (топ-10):\n"
    for char, percent in sorted_freq.items():
        result += f"{char}: {percent}%\n"
    return result

# ============= ШИФР ВИЖЕНЕРА =============
def vigenere_encrypt(text, key):
    if not key:
        return "Введите ключевое слово"
    text = text.upper()
    key = key.upper()
    result = ""
    key_index = 0
    for char in text:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - 65
            result += chr((ord(char) - 65 + shift) % 26 + 65)
            key_index += 1
        else:
            result += char
    return result

def vigenere_decrypt(text, key):
    if not key:
        return "Введите ключевое слово"
    text = text.upper()
    key = key.upper()
    result = ""
    key_index = 0
    for char in text:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - 65
            result += chr((ord(char) - 65 - shift) % 26 + 65)
            key_index += 1
        else:
            result += char
    return result

# ============= RSA С ВИЗУАЛИЗАЦИЕЙ ФАКТОРИЗАЦИИ =============
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def mod_inverse(e, phi):
    for d in range(2, phi):
        if (e * d) % phi == 1:
            return d
    return None

def factorize_with_steps(n, callback):
    factors = []
    temp_n = n
    i = 2
    while i * i <= temp_n:
        if temp_n % i == 0:
            factors.append(i)
            temp_n //= i
        else:
            i += 1 if i == 2 else 2
    if temp_n > 1:
        factors.append(temp_n)
    return factors

def generate_rsa_keys(p, q):
    if not (is_prime(p) and is_prime(q)):
        return None, "Оба числа должны быть простыми"
    if p == q:
        return None, "Числа должны быть разными"
    
    n = p * q
    phi = (p - 1) * (q - 1)
    
    e = 65537
    if e >= phi:
        e = 17
        while e < phi:
            if gcd(e, phi) == 1:
                break
            e += 2
    
    d = mod_inverse(e, phi)
    if d is None:
        return None, "Не удалось найти закрытый ключ"
    
    return (n, e, d), "Успешно"

# ============= СИМУЛЯТОР ЭНИГМЫ =============
class EnigmaSimulator:
    def __init__(self):
        # Стандартная проводка роторов (как в реальной Энигме)
        self.rotor1 = list("EKMFLGDQVZNTOWYHXUSPAIBRCJ")
        self.rotor2 = list("AJDKSIRUXBLHWTMCQGZNPYFVOE")
        self.rotor3 = list("BDFHJLCPRTXVZNYEIWGAKMUSQO")
        self.reflector = list("YRUHQSLDPXNGOKMIEBFZCWVJAT")
        
        self.rotor1_pos = 0
        self.rotor2_pos = 0
        self.rotor3_pos = 0
    
    def set_positions(self, pos1, pos2, pos3):
        self.rotor1_pos = pos1 % 26
        self.rotor2_pos = pos2 % 26
        self.rotor3_pos = pos3 % 26
    
    def rotate(self):
        """Поворот роторов (как одометр)"""
        self.rotor1_pos = (self.rotor1_pos + 1) % 26
        if self.rotor1_pos == 0:
            self.rotor2_pos = (self.rotor2_pos + 1) % 26
            if self.rotor2_pos == 0:
                self.rotor3_pos = (self.rotor3_pos + 1) % 26
    
    def encrypt_char(self, char):
        """Шифрование одного символа с текущими позициями"""
        if not char.isalpha():
            return char
        
        char = char.upper()
        idx = ord(char) - 65
        
        # Проход через роторы вперед (с учетом позиций)
        idx = (idx + self.rotor1_pos) % 26
        idx = ord(self.rotor1[idx]) - 65
        idx = (idx - self.rotor1_pos) % 26
        
        idx = (idx + self.rotor2_pos) % 26
        idx = ord(self.rotor2[idx]) - 65
        idx = (idx - self.rotor2_pos) % 26
        
        idx = (idx + self.rotor3_pos) % 26
        idx = ord(self.rotor3[idx]) - 65
        idx = (idx - self.rotor3_pos) % 26
        
        # Рефлектор
        idx = ord(self.reflector[idx]) - 65
        
        # Обратный проход через роторы
        idx = (idx + self.rotor3_pos) % 26
        idx = self.rotor3.index(chr(idx + 65))
        idx = (idx - self.rotor3_pos) % 26
        
        idx = (idx + self.rotor2_pos) % 26
        idx = self.rotor2.index(chr(idx + 65))
        idx = (idx - self.rotor2_pos) % 26
        
        idx = (idx + self.rotor1_pos) % 26
        idx = self.rotor1.index(chr(idx + 65))
        idx = (idx - self.rotor1_pos) % 26
        
        self.rotate()
        return chr(idx + 65)
    
    def encrypt_text(self, text):
        """Шифрование всего текста"""
        result = ""
        for char in text:
            result += self.encrypt_char(char)
        return result
        
class ECCSimulator:
    def __init__(self):
        # Параметры по умолчанию (учебная кривая)
        self.set_curve(23, 1, 1, (17, 20))
    
    def set_curve(self, p, a, b, G):
        """Установка параметров кривой"""
        self.p = p  # модуль (простое число)
        self.a = a  # коэффициент a
        self.b = b  # коэффициент b
        self.G = G  # базовая точка
    
    def mod_inverse(self, k, p):
        """Обратный элемент по модулю p"""
        for i in range(1, p):
            if (k * i) % p == 1:
                return i
        return None
    
    def point_double(self, point):
        """Удвоение точки на эллиптической кривой"""
        if point is None:
            return None
        
        x, y = point
        if y == 0:
            return None
        
        numerator = (3 * x * x + self.a) % self.p
        denominator = (2 * y) % self.p
        inv_denom = self.mod_inverse(denominator, self.p)
        
        if inv_denom is None:
            return None
        
        lam = (numerator * inv_denom) % self.p
        x3 = (lam * lam - 2 * x) % self.p
        y3 = (lam * (x - x3) - y) % self.p
        
        return (x3, y3)
    
    def point_add(self, point1, point2):
        """Сложение двух точек"""
        if point1 is None:
            return point2
        if point2 is None:
            return point1
        
        x1, y1 = point1
        x2, y2 = point2
        
        if x1 == x2 and y1 != y2:
            return None
        
        if x1 == x2 and y1 == y2:
            return self.point_double(point1)
        
        numerator = (y2 - y1) % self.p
        denominator = (x2 - x1) % self.p
        inv_denom = self.mod_inverse(denominator, self.p)
        
        if inv_denom is None:
            return None
        
        lam = (numerator * inv_denom) % self.p
        x3 = (lam * lam - x1 - x2) % self.p
        y3 = (lam * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_mult(self, k, point):
        """Умножение точки на скаляр"""
        result = None
        addend = point
        
        while k:
            if k & 1:
                result = self.point_add(result, addend)
            addend = self.point_double(addend)
            k >>= 1
        
        return result
    
    def generate_keypair(self, private_key):
        """Генерация пары ключей"""
        try:
            public_key = self.scalar_mult(private_key, self.G)
            return public_key
        except:
            return None
    
    def get_all_points(self):
        """Получение всех точек кривой"""
        points = []
        for x in range(self.p):
            for y in range(self.p):
                left = (y * y) % self.p
                right = (x * x * x + self.a * x + self.b) % self.p
                if left == right:
                    points.append((x, y))
        return points
    
    def is_point_on_curve(self, point):
        """Проверка, лежит ли точка на кривой"""
        if point is None:
            return False
        x, y = point
        left = (y * y) % self.p
        right = (x * x * x + self.a * x + self.b) % self.p
        return left == right
# ============= НАСТРОЙКА СТРАНИЦЫ =============
st.set_page_config(
    page_title="Криптография: интерактивная платформа",
    page_icon="🔐",
    layout="wide"
)

# ============= КАСТОМНЫЕ СТИЛИ =============
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e2e;
    }
    .main-header {
        color: #89b4fa;
        font-family: 'Cascadia Code', monospace;
        text-align: center;
        padding: 20px;
        font-size: 2.5em;
    }
    .sub-header {
        color: #cdd6f4;
        font-family: 'Cascadia Code', monospace;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #313244;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #45475a;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 8px 16px;
        font-family: 'Cascadia Code', monospace;
    }
    .stTabs [aria-selected="true"] {
        background-color: #89b4fa;
        color: #1e1e2e;
    }
    .result-box {
        background-color: #313244;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #89b4fa;
        color: #cdd6f4;
        font-family: 'Cascadia Code', monospace;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============= ЗАГОЛОВОК =============
st.markdown('<p class="main-header">🔐 Криптография: интерактивная платформа</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">От шифра Цезаря до эллиптических кривых</p>', unsafe_allow_html=True)

# ============= СОЗДАНИЕ ВКЛАДОК =============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📜 Шифр Цезаря", 
    "🔑 Шифр Виженера", 
    "🧮 RSA", 
    "⚙️ Энигма", 
    "📈 ECC"
])

# ============= ВКЛАДКА 1: ШИФР ЦЕЗАРЯ =============
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Шифрование и дешифрование")
        text_input = st.text_area("Введите текст:", "HELLO WORLD", height=150, key="caesar_input")
        shift = st.slider("Сдвиг (1-25):", 1, 25, 3)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔒 Зашифровать", use_container_width=True, key="caesar_encrypt_btn"):
                result = caesar_encrypt(text_input, shift)  # ваша функция
                st.session_state['caesar_result'] = result
        with col_btn2:
            if st.button("🔓 Расшифровать", use_container_width=True, key="caesar_decrypt_btn"):
                result = caesar_decrypt(text_input, shift)  # ваша функция
                st.session_state['caesar_result'] = result
        
        if 'caesar_result' in st.session_state:
            st.markdown(f'<div class="result-box">Результат:<br>{st.session_state["caesar_result"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Криптоанализ")
        brute_input = st.text_area("Текст для взлома:", "WKLVLVDVHFUHWPHVVDJH", height=100, key="brute_input")
        
        if st.button("⚡ Перебор всех сдвигов", use_container_width=True, key="caesar_bruteforce_btn"):
            results = caesar_bruteforce(brute_input)  # ваша функция
            st.session_state['brute_results'] = results
        
        if 'brute_results' in st.session_state:
            with st.expander("Результаты перебора"):
                st.text(st.session_state['brute_results'])
        
        st.subheader("Частотный анализ")
        freq_input = st.text_area("Текст для анализа:", "WKLVLVDVHFUHWPHVVDJH", height=100, key="freq_input")

        if st.button("📊 Анализировать частоты", use_container_width=True, key="caesar_freq_btn"):
            freq_result = frequency_analysis(freq_input)  # Это возвращает строку
            st.session_state['freq_result'] = freq_result

        if 'freq_result' in st.session_state:
            st.text(st.session_state['freq_result'])  # Просто показываем текст

# ============= ВКЛАДКА 2: ШИФР ВИЖЕНЕРА =============
with tab2:
    st.subheader("Шифр Виженера")
    col1, col2 = st.columns(2)
    
    with col1:
        vig_text = st.text_area("Введите текст:", "SECRET MESSAGE", height=150, key="vig_input")
        vig_key = st.text_input("Ключевое слово:", "KEY")
        
        if st.button("🔒 Зашифровать", use_container_width=True, key="vig_encrypt_btn"):
            result = vigenere_encrypt(vig_text, vig_key)  # ваша функция
            st.session_state['vig_result'] = result
    
    with col2:
        if st.button("🔓 Расшифровать", use_container_width=True, key="vig_decrypt_btn"):
            result = vigenere_decrypt(vig_text, vig_key)  # ваша функция
            st.session_state['vig_result'] = result
    
    if 'vig_result' in st.session_state:
        st.markdown(f'<div class="result-box">Результат:<br>{st.session_state["vig_result"]}</div>', unsafe_allow_html=True)

# ============= ВКЛАДКА 3: RSA =============
with tab3:
    st.subheader("RSA: Генерация ключей")
    col1, col2 = st.columns(2)
    
    with col1:
        p = st.number_input("Простое число p:", min_value=2, max_value=1000, value=61)
        q = st.number_input("Простое число q:", min_value=2, max_value=1000, value=53)
        
        if st.button("⚙️ Сгенерировать ключи", use_container_width=True, key="rsa_generate_btn"):
            keys, message = generate_rsa_keys(int(p), int(q))  # ваша функция
            if keys:
                st.session_state['rsa_keys'] = keys
                st.success(message)
            else:
                st.error(message)
    
    with col2:
        if 'rsa_keys' in st.session_state:
            st.markdown(f'<div class="result-box">'
                       f'Модуль n: {st.session_state["rsa_keys"][0]}<br>'
                       f'Открытый ключ: e={st.session_state["rsa_keys"][1]}<br>'
                       f'Закрытый ключ: d={st.session_state["rsa_keys"][2]}'
                       f'</div>', unsafe_allow_html=True)

# ============= ВКЛАДКА 4: ЭНИГМА =============
with tab4:
    st.subheader("Симулятор Энигмы")
    
    if 'enigma' not in st.session_state:
        st.session_state['enigma'] = EnigmaSimulator()  # ваш класс
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pos1 = st.number_input("Ротор 1:", 0, 25, 0)
    with col2:
        pos2 = st.number_input("Ротор 2:", 0, 25, 0)
    with col3:
        pos3 = st.number_input("Ротор 3:", 0, 25, 0)
    
    if st.button("Установить позиции", key="enigma_setpos_btn"):
        st.session_state['enigma'].set_positions(pos1, pos2, pos3)
        st.success(f"Позиции установлены: {pos1}, {pos2}, {pos3}")
    
    enigma_text = st.text_area("Введите текст:", "HELLO WORLD", height=150)
    
    if st.button("🔒 Зашифровать Энигмой", use_container_width=True, key="enigma_encrypt_btn"):
        st.session_state['enigma'].set_positions(pos1, pos2, pos3)
        result = st.session_state['enigma'].encrypt_text(enigma_text)
        st.session_state['enigma_result'] = result
        
    if 'enigma_result' in st.session_state:
        st.markdown("### Результат:")
        st.markdown(f'<div class="result-box">{st.session_state["enigma_result"]}</div>', 
                   unsafe_allow_html=True)
                   
# ============= ВКЛАДКА 5: ECC =============
with tab5:
    st.subheader("Криптография на эллиптических кривых")
    
    if 'ecc' not in st.session_state:
        st.session_state['ecc'] = ECCSimulator()
    
    # Настройки кривой
    with st.expander("⚙️ Настройки кривой", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_val = st.number_input("Модуль p (простое число):", 
                                   min_value=5, max_value=97, value=23, step=2, key="ecc_p")
        
        max_g = p_val - 1
        
        with col2:
            a_val = st.number_input("Коэффициент a:", 
                                   min_value=0, max_value=10, value=1, key="ecc_a")
        with col3:
            b_val = st.number_input("Коэффициент b:", 
                                   min_value=0, max_value=10, value=1, key="ecc_b")
        
        st.markdown("**Базовая точка G (x, y):**")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            default_gx = 17 if p_val > 17 else p_val // 2
            gx_val = st.number_input("Gx:", min_value=0, max_value=max_g, 
                                    value=default_gx, key="ecc_gx")
        with col_g2:
            default_gy = 20 if p_val > 20 else p_val // 2
            gy_val = st.number_input("Gy:", min_value=0, max_value=max_g, 
                                    value=default_gy, key="ecc_gy")
        
        if st.button("🔄 Применить новую кривую", key="ecc_apply_curve"):
            st.session_state['ecc'].set_curve(p_val, a_val, b_val, (gx_val, gy_val))
            if st.session_state['ecc'].is_point_on_curve((gx_val, gy_val)):
                st.success(f"Кривая обновлена: y² = x³ + {a_val}x + {b_val} mod {p_val}")
                if 'pub_key' in st.session_state:
                    del st.session_state['pub_key']
            else:
                st.error(f"Точка G({gx_val}, {gy_val}) не лежит на кривой!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Текущие параметры:**")
        st.markdown(f"Кривая: y² = x³ + {st.session_state['ecc'].a}x + {st.session_state['ecc'].b} mod {st.session_state['ecc'].p}")
        st.markdown(f"Базовая точка G: {st.session_state['ecc'].G}")
        
        priv_key = st.number_input("Приватный ключ:", 1, 50, 7, key="ecc_priv_key")
        
        if st.button("🔑 Сгенерировать публичный ключ", key="ecc_generate_btn"):
            pub_key = st.session_state['ecc'].generate_keypair(priv_key)
            if pub_key is not None:
                st.session_state['pub_key'] = pub_key
                st.success(f"Публичный ключ: {pub_key}")
            else:
                st.error("Не удалось сгенерировать ключ. Попробуйте другое число.")
    
    with col2:
        # 👇 ЭТА КНОПКА БЫЛА ПОТЕРЯНА - ВОТ ОНА:
        if st.button("📊 Показать все точки кривой", key="ecc_show_points_btn"):
            points = st.session_state['ecc'].get_all_points()
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1e1e2e')
            ax.set_facecolor('#313244')
            
            if points:
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                ax.scatter(x_vals, y_vals, c='#89b4fa', s=100, edgecolors='none', alpha=0.7)
            
            if st.session_state['ecc'].is_point_on_curve(st.session_state['ecc'].G):
                ax.scatter([st.session_state['ecc'].G[0]], [st.session_state['ecc'].G[1]], 
                          c='#a6e3a1', s=250, marker='s', edgecolors='none', 
                          label='Базовая точка G', zorder=5)
            
            if 'pub_key' in st.session_state and st.session_state['pub_key'] is not None:
                if st.session_state['ecc'].is_point_on_curve(st.session_state['pub_key']):
                    ax.scatter([st.session_state['pub_key'][0]], [st.session_state['pub_key'][1]], 
                              c='#f38ba8', s=300, marker='*', edgecolors='none', 
                              label='Публичный ключ', zorder=6)
            
            ax.set_xlabel('x', color='#cdd6f4', fontsize=12)
            ax.set_ylabel('y', color='#cdd6f4', fontsize=12)
            ax.set_title('Точки эллиптической кривой', color='#cdd6f4', pad=20, fontsize=14)
            ax.grid(True, alpha=0.2, color='#45475a', linestyle='--')
            
            for spine in ax.spines.values():
                spine.set_color('#45475a')
                spine.set_linewidth(1)
            
            ax.tick_params(colors='#cdd6f4', grid_color='#45475a')
            ax.set_xlim(-1, st.session_state['ecc'].p)
            ax.set_ylim(-1, st.session_state['ecc'].p)
            
            if points:
                legend = ax.legend(facecolor='#313244', labelcolor='#cdd6f4', 
                                  edgecolor='#45475a', framealpha=1, loc='upper right')
                legend.get_frame().set_linewidth(1)
            
            st.pyplot(fig)

# ============= ЗАПУСК =============
if __name__ == "__main__":
    # Запускать через терминал: streamlit run web_app.py
    pass