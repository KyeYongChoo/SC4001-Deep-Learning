import requests
import socket
import time

def check_host(url, name=None, timeout=5):
	try:
		start = time.time()
		r = requests.head(url, timeout=timeout)
		elapsed = (time.time() - start) * 1000
		print(f"✅ {name or url} reachable — status {r.status_code}, {elapsed:.1f} ms")
		return True
	except requests.exceptions.RequestException as e:
		print(f"❌ {name or url} blocked or unreachable — {e.__class__.__name__}: {e}")
		return False

def resolve_dns(host):
	try:
		ip = socket.gethostbyname(host)
		print(f"🧭 {host} resolves to {ip}")
		return True
	except Exception as e:
		print(f"⚠️ Could not resolve {host}: {e}")
		return False

def hugging_face_connectivity_test():
	print("=== Hugging Face Connectivity Test ===\n")

	# Step 1. DNS resolution
	huggingface_dns = resolve_dns("huggingface.co")
	cdn_dns = resolve_dns("cdn-lfs.huggingface.co")

	# Step 2. HTTPS access
	print("\n--- Checking HTTPS endpoints ---")
	huggingface_https = check_host("https://huggingface.co", name="huggingface.co")
	cdn_https = check_host("https://cdn-lfs.huggingface.co", name="cdn-lfs.huggingface.co")

	# Step 3. try a real model file HEAD request
	print("\n--- Checking model file availability ---")
	head_req = check_host("https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors",
			name="ViT base patch16 weights")

	print("\n✅ Test complete.\nIf any line shows ❌, your Python environment cannot reach Hugging Face.\n"
		"Try a full VPN (system-wide), set HTTPS_PROXY, or download weights manually.")
	
	return huggingface_dns and cdn_dns and huggingface_https and cdn_https and head_req
