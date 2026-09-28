# Builds the llm-obs collector as a small, static-ish binary and runs it in a
# minimal base image. Ollama/vLLM are NOT part of this image: they run on the
# developer's host as part of their own local-LLM setup, and the collector
# reaches them via LLM_OBS_OLLAMA_URL / LLM_OBS_VLLM_METRICS_URL.
FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/llm-obs-collector ./collector/cmd/collector

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/llm-obs-collector /usr/local/bin/llm-obs-collector
VOLUME ["/data"]
ENV LLM_OBS_DB=/data/llm-obs.db
ENV LLM_OBS_LISTEN=:8765
EXPOSE 8765
ENTRYPOINT ["/usr/local/bin/llm-obs-collector"]
