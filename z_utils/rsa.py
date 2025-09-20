from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import base64


private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

public_key = private_key.public_key()

SIGN_DATA = b"LICENSE_v1_20280909"

signature = private_key.sign(SIGN_DATA, padding.PKCS1v15(), hashes.SHA256())

# 输出硬编码用的公钥和签名
PUBLIC_KEY_PEM = (
    public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode("utf-8")
    .strip()
)
SIGNATURE_B64 = base64.b64encode(signature).decode("utf-8")

print("=== 请将以下内容复制到你的程序中 ===")
print(PUBLIC_KEY_PEM)
print(f"SIGNATURE_B64 = '{SIGNATURE_B64}'")
print("=====================================")

# 本地验证
try:
    public_key.verify(
        base64.b64decode(SIGNATURE_B64), SIGN_DATA, padding.PKCS1v15(), hashes.SHA256()
    )
    print("✅ 本地验证成功，生成的常量是正确的。")
except Exception as e:
    print(f"❌ 本地验证失败: {e}")


def _verify_integrity_and_get_end_date():

    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        import base64
        from datetime import datetime

        PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3aUZxvR67EI+i45F7uQ/
7WlT8w5kDfdiQG4tD7jKgVuY5gZvoDehquk3mYy2eAVGVfgeAGod9le8hfrEFIcW
+Ba7WMm5Fz3Oweng5o60T8td+4baTDwUXP8QPTV6EgQvzbgYEPUT+YHw0ELYsIah
nz5kZ6LQQUyT3RJQIoAaXuJI4gINbMWi/Ss0c2L0C7vF5oHyIV4/vsNOJ/QaL1Ff
QxFUX2Z+tO6Fd9gLM+JizHCthQDHXmTD4R11azd/VjWZps9wyhSALZ2OH7wf224g
IAcaiG+P2Cw8kuOpyGSZjHAnVHttsfGNtgbkhP//GQTB0L0onEmEOKyLJf1rZHIg
+wIDAQAB
-----END PUBLIC KEY-----"""
        SIGNATURE_B64 = "bse1O3bY28BnrA55ynF2odv9rcFBBsp16XxT59o99UiAjU4OSJAtggmfInMn9kq/SkxNKdLKhVe9j1Q/eytA4vSF6MOIA2iIit/6mG56O56gjgJrSDfvN/ol8VXG4MbWb0/+ZqRbhYgRwFWZTazq3LS1iXYK7WAoiDvR2JouyTa0yVjkg73S7gA68jTJblS1K0y5w3lDNYEzTBiZBOLGiJskiTZaSCL27HonzOb3VYXs+Mi2o0HfwbBRjGUnSPi4i4IYQcVgUZN4zKyCtqt/Ix+l2tf7N08LfJaBh0BtXqDpXJFnG0VXSZ8+nwG0cZHpCvaomyDGGle+rpDL0/J2hQ=="
        pub_key = serialization.load_pem_public_key(PUBLIC_KEY_PEM.encode("utf-8"))
        sig = base64.b64decode(SIGNATURE_B64)
        data_to_verify = b"LICENSE_v1_20280909"
        pub_key.verify(sig, data_to_verify, padding.PKCS1v15(), hashes.SHA256())
        end_date_str = data_to_verify.decode("utf-8").split("_")[-1]
        current_date_str = datetime.now().strftime("%Y%m%d")
        return current_date_str < end_date_str
    except Exception:
        return False


if __name__ == "__main__":
    """
    uv run z_utils/rsa.py
    """
    is_valid = _verify_integrity_and_get_end_date()
    print(f"授权状态: {'有效' if is_valid else '已过期或无效'}")
