#!/bin/bash
# Script để tạo và setup môi trường ảo Python

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "=========================================="
echo "Setup môi trường ảo cho Model Extraction"
echo "=========================================="
echo ""

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Không tìm thấy python3. Vui lòng cài đặt Python 3.8 trở lên."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python version: $(python3 --version)"

# Tạo virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Môi trường ảo đã tồn tại tại $VENV_DIR"
    read -p "Bạn có muốn xóa và tạo lại? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Đang xóa môi trường ảo cũ..."
        rm -rf "$VENV_DIR"
    else
        echo "ℹ️  Sử dụng môi trường ảo hiện có"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Đang tạo môi trường ảo..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Đã tạo môi trường ảo tại $VENV_DIR"
fi

# Kích hoạt virtual environment
echo "🔄 Đang kích hoạt môi trường ảo..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Đang nâng cấp pip..."
pip install --upgrade pip setuptools wheel

# Cài đặt dependencies
echo "📥 Đang cài đặt dependencies từ requirements.txt..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
    echo "✅ Đã cài đặt dependencies từ requirements.txt"
else
    echo "⚠️  Không tìm thấy requirements.txt"
fi

# Cài đặt dependencies cho ember trước
echo "📥 Đang cài đặt dependencies cho ember (tqdm, lief)..."
pip install tqdm lief

# Cài đặt ember từ GitHub (không có trên PyPI)
echo "📥 Đang cài đặt ember từ GitHub..."
pip install git+https://github.com/endgameinc/ember.git
echo "✅ Đã cài đặt ember"

echo ""
echo "=========================================="
echo "✅ Setup hoàn tất!"
echo "=========================================="
echo ""
echo "Để kích hoạt môi trường ảo, chạy lệnh:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Hoặc sử dụng:"
echo "  source venv/bin/activate"
echo ""
echo "Để tắt môi trường ảo:"
echo "  deactivate"
echo ""

