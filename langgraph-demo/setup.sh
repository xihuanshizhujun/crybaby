#!/bin/bash
# 快速设置脚本

echo "🚀 开始配置项目..."

# 1. 创建 .env 文件（如果不存在）
if [ ! -f .env ]; then
    echo "📝 创建 .env 文件..."
    cp env.example .env
    echo "✅ .env 文件已创建，请编辑并填入你的配置"
else
    echo "✅ .env 文件已存在"
fi

# 2. 创建 uploads 目录
if [ ! -d uploads ]; then
    echo "📁 创建 uploads 目录..."
    mkdir -p uploads
    echo "✅ uploads 目录已创建"
else
    echo "✅ uploads 目录已存在"
fi

# 3. 检查依赖是否安装
echo "📦 检查依赖..."
if ! python -c "import agent" 2>/dev/null; then
    echo "⚠️  依赖未安装，请运行: uv add . --dev"
else
    echo "✅ 依赖已安装"
fi

echo ""
echo "✨ 配置完成！"
echo ""
echo "📋 下一步："
echo "1. 编辑 .env 文件，填入你的配置"
echo "2. 启动向量数据库（Docker）："
echo "   docker run -d --name weaviate -p 8080:8080 -p 50051:50051 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:latest"
echo "3. 启动项目："
echo "   streamlit run streamlit_app.py"


