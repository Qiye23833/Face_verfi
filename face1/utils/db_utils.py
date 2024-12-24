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
        
        try:
            # 检查表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # 如果表不存在，创建新表
                cursor.execute('''
                CREATE TABLE faces (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    gender TEXT,
                    position TEXT,
                    department TEXT,
                    person_type TEXT,
                    entry_date TEXT,
                    feature_vector BLOB NOT NULL,
                    face_image BLOB,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
            else:
                # 如果表存在，检查并添加缺失的列
                cursor.execute('PRAGMA table_info(faces)')
                existing_columns = [column[1] for column in cursor.fetchall()]
                
                # 需要添加的新列
                new_columns = {
                    'gender': 'TEXT',
                    'position': 'TEXT',
                    'department': 'TEXT',
                    'person_type': 'TEXT',
                    'entry_date': 'TEXT'
                }
                
                # 添加缺失的列
                for column_name, column_type in new_columns.items():
                    if column_name not in existing_columns:
                        cursor.execute(f'ALTER TABLE faces ADD COLUMN {column_name} {column_type}')
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"Database error: {str(e)}")
            raise Exception("数据库初始化失败")
    
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
        将输入的人脸特征与数据库中特征进行匹配
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
    
    def update_name(self, face_id, new_name):
        """更新人脸姓名"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE faces SET name = ? WHERE id = ?', (new_name, face_id))
        self.conn.commit()
    
    def update_id(self, old_id, new_id):
        """更新人脸ID"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE faces SET id = ? WHERE id = ?', (new_id, old_id))
        self.conn.commit()
    
    def id_exists(self, face_id):
        """检查ID是否已存在"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM faces WHERE id = ?', (face_id,))
        return cursor.fetchone()[0] > 0
    
    def add_empty_face(self, face_id, name):
        """添加新的空人脸记录"""
        cursor = self.conn.cursor()
        # 使用空的特征向量
        empty_features = np.zeros(128)  # 使用128维的零向量
        cursor.execute(
            'INSERT INTO faces (id, name, feature_vector) VALUES (?, ?, ?)',
            (face_id, name, pickle.dumps(empty_features))
        )
        self.conn.commit()
    
    def add_face_with_id(self, id, name, feature_vector, face_image=None):
        """
        添加带指定ID的人脸信息到数据库
        Args:
            id: 指定的ID
            name: 人名
            feature_vector: 人脸特征向量
            face_image: 人脸图像数据
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO faces (id, name, feature_vector, face_image) VALUES (?, ?, ?, ?)',
            (id, name, pickle.dumps(feature_vector), face_image)
        )
        self.conn.commit()
    
    def add_face_with_info(self, info, feature_vector, face_image=None):
        """
        添加带完整信息的人脸到数据库
        Args:
            info: 包含人脸信息的字典
            feature_vector: 人脸特征向量
            face_image: 人脸图像数据
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                '''INSERT INTO faces (
                    id, name, gender, position, department, 
                    person_type, entry_date, feature_vector, face_image
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    info['id'], info['name'], info['gender'], 
                    info['position'], info['department'], info['type'],
                    info['entry_date'], pickle.dumps(feature_vector), face_image
                )
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {str(e)}")
            raise Exception("数据库操作失败")
        except Exception as e:
            print(f"Error saving face: {str(e)}")
            raise Exception("保存人脸信息失败")
    
    def get_all_faces_with_info(self):
        """获取���有已注册的人脸完整信息"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, name, gender, position, department, 
                   person_type, entry_date, feature_vector, face_image, create_time 
            FROM faces
        ''')
        results = cursor.fetchall()
        
        faces = []
        for row in results:
            face_info = {
                'id': row[0],
                'name': row[1],
                'gender': row[2] or '',
                'position': row[3] or '',
                'department': row[4] or '',
                'person_type': row[5] or '',
                'entry_date': row[6] or '',
                'feature_vector': pickle.loads(row[7]),
                'face_image': row[8],
                'create_time': row[9]
            }
            faces.append(face_info)
        
        return faces
    
    def get_face_info(self, face_id):
        """获取指定ID的人脸完整信息"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, name, gender, position, department, 
                   person_type, entry_date, feature_vector, face_image, create_time 
            FROM faces 
            WHERE id = ?
        ''', (face_id,))
        row = cursor.fetchone()
        
        if row:
            face_info = {
                'id': row[0],
                'name': row[1],
                'gender': row[2] or '',
                'position': row[3] or '',
                'department': row[4] or '',
                'person_type': row[5] or '',
                'entry_date': row[6] or '',
                'feature_vector': pickle.loads(row[7]),
                'face_image': row[8],
                'create_time': row[9]
            }
            return face_info
        return None