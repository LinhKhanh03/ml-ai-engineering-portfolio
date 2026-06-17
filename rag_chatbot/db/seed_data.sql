CREATE TABLE nganh (
    id INT PRIMARY KEY IDENTITY(1,1),
    ten_nganh NVARCHAR(200) NOT NULL
);

CREATE TABLE chuyen_nganh (
    id INT PRIMARY KEY IDENTITY(1,1),
    nganh_id INT NOT NULL,
    ten_chuyen_nganh NVARCHAR(200) NOT NULL,
    FOREIGN KEY (nganh_id) REFERENCES nganh(id)
);

CREATE TABLE mon_hoc (
    id INT PRIMARY KEY IDENTITY(1,1),
    chuyen_nganh_id INT NOT NULL,
    ten_mon NVARCHAR(200) NOT NULL,
    so_tin_chi INT NOT NULL,
    FOREIGN KEY (chuyen_nganh_id) REFERENCES chuyen_nganh(id)
);

INSERT INTO nganh (ten_nganh) VALUES
(N'Hệ thống thông tin quản lý'),
(N'Tài chính - Ngân hàng');

INSERT INTO chuyen_nganh (nganh_id, ten_chuyen_nganh) VALUES
(1, N'Công nghệ Fintech'),
(1, N'Hệ thống thông tin doanh nghiệp'),
(2, N'Ngân hàng');

INSERT INTO mon_hoc (chuyen_nganh_id, ten_mon, so_tin_chi) VALUES
(1, N'Nhập môn học máy', 3),
(1, N'Cơ sở dữ liệu', 3),
(1, N'Phân tích dữ liệu kinh doanh', 3),
(2, N'Hệ thống thông tin doanh nghiệp', 3),
(2, N'Quản trị dự án CNTT', 2),
(3, N'Nghiệp vụ ngân hàng thương mại', 3),
(3, N'Quản trị rủi ro tài chính', 3);