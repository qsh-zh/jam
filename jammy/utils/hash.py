import hashlib
import pickle

__all__ = ["md5_encodef", "md5_encode_obj"]

def md5_encodef(f_input):
    file_code = open(f_input, "rb").read()
    md5_hash = hashlib.md5()
    md5_hash.update(file_code)
    digest = md5_hash.hexdigest()
    return digest

def md5_encode_obj(f_obj):
    byte_obj = pickle.dumps(f_obj)
    md5_hash = hashlib.md5()
    md5_hash.update(byte_obj)
    digest = md5_hash.hexdigest()
    return digest
