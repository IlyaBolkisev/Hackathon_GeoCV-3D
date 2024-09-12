# nginx/Dockerfile
FROM nginx:alpine

COPY deploy/nginx/nginx.conf /etc/nginx/nginx.conf
COPY static /usr/share/nginx/html/static
COPY static/index.html /usr/share/nginx/html/index.html
