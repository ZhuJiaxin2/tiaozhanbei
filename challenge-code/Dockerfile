# 使用 Node.js 官方镜像
FROM node:18

# 创建工作目录
WORKDIR /app

# 复制 package.json 和 yarn.lock
COPY package.json yarn.lock ./

# 安装依赖
RUN yarn install

# 复制所有文件到工作目录
COPY . .

# 暴露应用运行端口（根据你的应用需要，假设是3000）
EXPOSE 3000

# 运行应用
CMD ["yarn", "start"]
