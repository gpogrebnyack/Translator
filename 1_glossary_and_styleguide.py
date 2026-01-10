#!/usr/bin/env python3
"""
Скрипт для автоматической генерации глоссария и стилистического гайда
для перевода книги с использованием Gemini API через OpenRouter.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
import time

# Загрузка переменных окружения
load_dotenv()

# Константы
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'google/gemini-3-pro-preview')
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Цвета для вывода в консоль
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_step(message):
    """Вывод информации о текущем шаге"""
    print(f"{Colors.OKBLUE}▶ {message}{Colors.ENDC}")


def print_success(message):
    """Вывод сообщения об успехе"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Вывод сообщения об ошибке"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_warning(message):
    """Вывод предупреждения"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def load_config(config_path='config.json'):
    """Загрузка конфигурации из JSON файла"""
    print_step(f"Загрузка конфигурации из {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print_success("Конфигурация загружена")
        return config
    except FileNotFoundError:
        print_error(f"Файл конфигурации {config_path} не найден")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Ошибка парсинга JSON: {e}")
        sys.exit(1)


def load_book(book_path):
    """Загрузка текста книги из JSON файла"""
    print_step(f"Загрузка книги из {book_path}")
    try:
        with open(book_path, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
        
        # Подсчет глав в зависимости от структуры
        total_chapters = 0
        if 'parts' in book_data:
            # Структура с частями (Locklands)
            for part in book_data.get('parts', []):
                total_chapters += len(part.get('chapters', []))
            print_success(f"Книга загружена: {len(book_data.get('parts', []))} частей, {total_chapters} глав")
        elif 'chapters' in book_data:
            # Структура с главами без частей (Katabasis)
            total_chapters = len(book_data.get('chapters', []))
            print_success(f"Книга загружена: {total_chapters} глав")
        elif isinstance(book_data, list):
            # Старая структура - просто массив глав
            total_chapters = len(book_data)
            print_success(f"Книга загружена: {total_chapters} глав")
        else:
            print_error("Неизвестная структура JSON файла")
            sys.exit(1)
        
        return book_data
    except FileNotFoundError:
        print_error(f"Файл книги {book_path} не найден")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Ошибка парсинга JSON книги: {e}")
        sys.exit(1)


def load_prompt(prompt_path):
    """Загрузка промпта из markdown файла"""
    print_step(f"Загрузка промпта из {prompt_path}")
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        print_success("Промпт загружен")
        return prompt
    except FileNotFoundError:
        print_error(f"Файл промпта {prompt_path} не найден")
        sys.exit(1)


def prepare_book_text(book_data, max_chars=500000):
    """
    Подготовка текста книги для отправки в API.
    Ограничивает размер текста для избежания превышения лимитов API.
    """
    print_step("Подготовка текста книги")
    
    text_parts = []
    total_chars = 0
    
    # Проверяем структуру файла
    if 'parts' in book_data:
        # Структура с частями (Locklands)
        for part in book_data.get('parts', []):
            part_title = part.get('part_title', 'Без названия')
            text_parts.append(f"\n\n# {part_title}\n\n")
            
            for chapter in part.get('chapters', []):
                chapter_title = chapter.get('title', 'Без названия')
                text_parts.append(f"\n## {chapter_title}\n\n")
                
                content = chapter.get('content', '')
                if total_chars + len(content) > max_chars:
                    print_warning(f"Достигнут лимит символов ({max_chars}). Используется частичный текст.")
                    break
                text_parts.append(content + '\n')
                total_chars += len(content)
            
            if total_chars > max_chars:
                break
                
    elif 'chapters' in book_data:
        # Структура с главами без частей (Katabasis)
        for chapter in book_data.get('chapters', []):
            chapter_title = chapter.get('title', 'Без названия')
            text_parts.append(f"\n\n## {chapter_title}\n\n")
            
            content = chapter.get('content', '')
            if total_chars + len(content) > max_chars:
                print_warning(f"Достигнут лимит символов ({max_chars}). Используется частичный текст.")
                break
            text_parts.append(content + '\n')
            total_chars += len(content)
            
    elif isinstance(book_data, list):
        # Старая структура - просто массив глав с параграфами
        for chapter in book_data:
            chapter_title = chapter.get('title', 'Без названия')
            text_parts.append(f"\n\n## {chapter_title}\n\n")
            
            for paragraph in chapter.get('paragraphs', []):
                para_text = paragraph.get('text', '')
                if total_chars + len(para_text) > max_chars:
                    print_warning(f"Достигнут лимит символов ({max_chars}). Используется частичный текст.")
                    break
                text_parts.append(para_text + '\n')
                total_chars += len(para_text)
            
            if total_chars > max_chars:
                break
    
    full_text = ''.join(text_parts)
    print_success(f"Подготовлено {total_chars} символов текста")
    return full_text


def call_openrouter_api(system_prompt, user_message, temperature=0.7, max_retries=3):
    """
    Отправка запроса к OpenRouter API с повторными попытками при ошибках
    """
    if not OPENROUTER_API_KEY:
        print_error("OPENROUTER_API_KEY не найден в .env файле")
        sys.exit(1)
    
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/translator-app',
        'X-Title': 'Book Translator - Glossary & Style Guide Generator'
    }
    
    payload = {
        'model': OPENROUTER_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ],
        'temperature': temperature,
        'max_tokens': 16000
    }
    
    for attempt in range(max_retries):
        try:
            print_step(f"Отправка запроса к API (попытка {attempt + 1}/{max_retries})")
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=300  # 5 минут таймаут
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print_success("Ответ получен от API")
                return content
            else:
                print_error(f"Ошибка API: {response.status_code}")
                print_error(f"Ответ: {response.text}")
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print_warning(f"Повторная попытка через {wait_time} секунд...")
                    time.sleep(wait_time)
                else:
                    print_error("Исчерпаны все попытки")
                    sys.exit(1)
                    
        except requests.exceptions.Timeout:
            print_error("Превышено время ожидания ответа от API")
            if attempt < max_retries - 1:
                print_warning("Повторная попытка...")
                time.sleep(10)
            else:
                sys.exit(1)
                
        except Exception as e:
            print_error(f"Ошибка при запросе к API: {e}")
            if attempt < max_retries - 1:
                print_warning("Повторная попытка...")
                time.sleep(10)
            else:
                sys.exit(1)


def extract_json_from_response(response_text):
    """
    Извлечение JSON из ответа API (может быть обернут в markdown блок)
    """
    # Попытка найти JSON в markdown блоке
    if '```json' in response_text:
        start = response_text.find('```json') + 7
        end = response_text.find('```', start)
        json_text = response_text[start:end].strip()
    elif '```' in response_text:
        start = response_text.find('```') + 3
        end = response_text.find('```', start)
        json_text = response_text[start:end].strip()
    else:
        json_text = response_text.strip()
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Если не удалось распарсить, возвращаем как есть
        print_warning("Не удалось извлечь JSON, сохраняем как текст")
        return None


def generate_glossary(book_text, prompt_glossary):
    """
    Генерация глоссария на основе текста книги
    """
    print(f"\n{Colors.HEADER}{'='*60}")
    print("ГЕНЕРАЦИЯ ГЛОССАРИЯ")
    print(f"{'='*60}{Colors.ENDC}\n")
    
    system_prompt = "Ты — ведущий редактор и специалист по локализации художественной литературы."
    
    user_message = f"{prompt_glossary}\n\n# ТЕКСТ КНИГИ:\n\n{book_text}"
    
    response = call_openrouter_api(system_prompt, user_message, temperature=0.5)
    
    # Попытка извлечь JSON
    glossary_json = extract_json_from_response(response)
    
    if glossary_json:
        return glossary_json
    else:
        # Если не удалось извлечь JSON, возвращаем текст
        return response


def generate_styleguide(book_text, glossary_text, prompt_styleguide):
    """
    Генерация стилистического гайда на основе текста книги и глоссария
    """
    print(f"\n{Colors.HEADER}{'='*60}")
    print("ГЕНЕРАЦИЯ СТИЛИСТИЧЕСКОГО ГАЙДА")
    print(f"{'='*60}{Colors.ENDC}\n")
    
    system_prompt = "Ты — эксперт по литературному переводу и стилистическому анализу текстов."
    
    # Подготовка глоссария для включения в промпт
    if isinstance(glossary_text, dict):
        glossary_str = json.dumps(glossary_text, ensure_ascii=False, indent=2)
    else:
        glossary_str = str(glossary_text)
    
    user_message = f"{prompt_styleguide}\n\n# ГЛОССАРИЙ:\n\n```json\n{glossary_str}\n```\n\n# ТЕКСТ КНИГИ:\n\n{book_text}"
    
    response = call_openrouter_api(system_prompt, user_message, temperature=0.7)
    
    return response


def save_glossary(glossary, output_path):
    """
    Сохранение глоссария в файл
    """
    print_step(f"Сохранение глоссария в {output_path}")
    
    # Создание директории если не существует
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(glossary, dict):
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(glossary))
        print_success(f"Глоссарий сохранен: {output_path}")
    except Exception as e:
        print_error(f"Ошибка при сохранении глоссария: {e}")
        sys.exit(1)


def save_styleguide(styleguide, output_path):
    """
    Сохранение стилистического гайда в файл
    """
    print_step(f"Сохранение стилистического гайда в {output_path}")
    
    # Создание директории если не существует
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(styleguide)
        print_success(f"Стилистический гайд сохранен: {output_path}")
    except Exception as e:
        print_error(f"Ошибка при сохранении стилистического гайда: {e}")
        sys.exit(1)


def main():
    """
    Основная функция скрипта
    """
    from config_helper import get_paths
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   ГЕНЕРАТОР ГЛОССАРИЯ И СТИЛИСТИЧЕСКОГО ГАЙДА ДЛЯ КНИГИ   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    # 1. Загрузка конфигурации
    try:
        paths = get_paths()
        paths.ensure_work_dir()
    except FileNotFoundError as e:
        print_error(f"Ошибка загрузки конфигурации: {e}")
        sys.exit(1)
    
    print_step(f"Обработка книги: {paths.book}")
    print_step(f"Входной файл: {paths.output_json}")
    print_step(f"Глоссарий: {paths.glossary_json}")
    print_step(f"Стилистический гайд: {paths.styleguide_md}")
    
    # 2. Загрузка книги
    book_data = load_book(str(paths.output_json))
    book_text = prepare_book_text(book_data, max_chars=paths.glossary_max_chars)
    
    # 3. Загрузка промптов
    prompt_glossary = load_prompt('.agent/Workflows/prompt_glossary.md')
    prompt_styleguide = load_prompt('.agent/Workflows/prompt_styleguide.md')
    
    # 4. Генерация глоссария
    glossary = generate_glossary(book_text, prompt_glossary)
    save_glossary(glossary, str(paths.glossary_json))
    
    # 5. Генерация стилистического гайда
    styleguide = generate_styleguide(book_text, glossary, prompt_styleguide)
    save_styleguide(styleguide, str(paths.styleguide_md))
    
    # 6. Финальное сообщение
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                    УСПЕШНО ЗАВЕРШЕНО!                      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    print_success(f"Глоссарий: {paths.glossary_json}")
    print_success(f"Стилистический гайд: {paths.styleguide_md}")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Прервано пользователем{Colors.ENDC}\n")
        sys.exit(0)
    except Exception as e:
        print_error(f"Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
