import sqlite3
import numpy as np
import pickle

class FaceDatabase:
    """
    人脸数据库管理类
    用于存储和检索人脸信息
    """
    def __init__(self, db_path='face_db.sqlite'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        """创建必要的数据表"""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            feature_vector BLOB NOT NULL,
            face_image BLOB,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()
        
    def add_face(self, name, feature_vector, face_image=None):
        """
        添加人脸信息到数据库
        Args:
            name: 人名
            feature_vector: 人脸特征向量
            face_image: 人脸图像数据
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO faces (name, feature_vector, face_image) VALUES (?, ?, ?)',
            (name, pickle.dumps(feature_vector), face_image)
        )
        self.conn.commit()
        
    def get_all_faces(self):
        """获取所有已注册的人脸信息"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, name, feature_vector FROM faces')
        results = cursor.fetchall()
        return [(id, name, pickle.loads(fv)) for id, name, fv in results]
    
    def get_all_faces_with_images(self):
        """获取所有已注册的人脸信息（包括图像）"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, name, feature_vector, face_image FROM faces')
        results = cursor.fetchall()
        return [(id, name, pickle.loads(fv), img) for id, name, fv, img in results]
    
    def delete_face(self, face_id):
        """删除指定的人脸信息"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        self.conn.commit()
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()
    
    def match_face(self, feature_vector, threshold=0.6):
        """
        将输入的人脸特征与数据库中的特征进行匹配
        Args:
            feature_vector: 输入的人脸特征向量
            threshold: 匹配阈值，越小越严格
        Returns:
            (id, name, similarity) 或 None: 匹配成功返回信息，失败返回None
        """
        faces = self.get_all_faces()
        if not faces:
            return None
        
        # 计算与所有已注册人脸的相似度
        max_similarity = 0
        best_match = None
        
        for id, name, db_feature in faces:
            # 计算余弦相似度
            similarity = np.dot(feature_vector, db_feature) / (
                np.linalg.norm(feature_vector) * np.linalg.norm(db_feature))
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = (id, name, similarity)
        
        # 如果最佳匹配的相似度超过阈值，返回匹配结果
        if max_similarity >= threshold:
            return best_match
        return None 