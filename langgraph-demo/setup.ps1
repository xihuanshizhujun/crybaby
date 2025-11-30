# PowerShell 快速设置脚本

Write-Host "🚀 开始配置项目..." -ForegroundColor Green

# 1. 创建 .env 文件（如果不存在）
if (-not (Test-Path .env)) {
    Write-Host "📝 创建 .env 文件..." -ForegroundColor Yellow
    Copy-Item env.example .env
    Write-Host "✅ .env 文件已创建，请编辑并填入你的配置" -ForegroundColor Green
} else {
    Write-Host "✅ .env 文件已存在" -ForegroundColor Green
}

# 2. 创建 uploads 目录
if (-not (Test-Path uploads)) {
    Write-Host "📁 创建 uploads 目录..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path uploads | Out-Null
    Write-Host "✅ uploads 目录已创建" -ForegroundColor Green
} else {
    Write-Host "✅ uploads 目录已存在" -ForegroundColor Green
}

# 3. 检查依赖是否安装
Write-Host "📦 检查依赖..." -ForegroundColor Yellow
try {
    python -c "import agent" 2>$null
    Write-Host "✅ 依赖已安装" -ForegroundColor Green
} catch {
    Write-Host "⚠️  依赖未安装，请运行: uv add . --dev" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✨ 配置完成！" -ForegroundColor Green
Write-Host ""
Write-Host "📋 下一步：" -ForegroundColor Cyan
Write-Host "1. 编辑 .env 文件，填入你的配置"
Write-Host "2. 启动向量数据库（Docker）："
Write-Host "   docker run -d --name weaviate -p 8080:8080 -p 50051:50051 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:latest"
Write-Host "3. 启动项目："
Write-Host "   streamlit run streamlit_app.py"


