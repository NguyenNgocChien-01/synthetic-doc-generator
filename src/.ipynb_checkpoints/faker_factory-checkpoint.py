"""
Module sinh dữ liệu văn bản giả sử dụng thư viện Faker.
Ánh xạ từng loại trường trong base.json sang hàm Faker tương ứng.
"""

import random
import string
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from faker import Faker

logger = logging.getLogger(__name__)


class FakerFactory:
    """
    Lớp sinh dữ liệu giả đa dạng từ thư viện Faker.

    Hỗ trợ nhiều kiểu trường dữ liệu và nhiều locale ngôn ngữ khác nhau,
    phục vụ việc tạo nội dung phù hợp với từng loại tài liệu.
    """

    # Bảng ánh xạ tên kiểu faker sang phương thức xử lý
    FAKER_TYPE_HANDLERS = {
        # Thông tin cá nhân
        "full_name": "_gen_full_name",
        "first_name": "_gen_first_name",
        "last_name": "_gen_last_name",
        "gender": "_gen_gender",
        "date_of_birth": "_gen_date_of_birth",
        "nationality": "_gen_nationality",
        "country": "_gen_country",
        "country_code": "_gen_country_code",
        # Số tài liệu
        "passport_number": "_gen_passport_number",
        "id_number": "_gen_id_number",
        "license_number": "_gen_license_number",
        "medicare_number": "_gen_medicare_number",
        "account_number": "_gen_account_number",
        # Ngày tháng
        "issue_date": "_gen_issue_date",
        "expiry_date": "_gen_expiry_date",
        "date_formatted": "_gen_date_formatted",
        # Thông tin liên lạc và địa chỉ
        "address": "_gen_address",
        "street_address": "_gen_street_address",
        "city": "_gen_city",
        "state": "_gen_state",
        "postcode": "_gen_postcode",
        "phone_number": "_gen_phone_number",
        "email": "_gen_email",
        # Thông tin tổ chức
        "company": "_gen_company",
        "company_abn": "_gen_company_abn",
        # Dữ liệu MRZ cho hộ chiếu
        "mrz_line1": "_gen_mrz_line1",
        "mrz_line2": "_gen_mrz_line2",
        # Dữ liệu tiện ích
        "utility_account_number": "_gen_utility_account_number",
        "meter_reading": "_gen_meter_reading",
        "amount_due": "_gen_amount_due",
        "billing_period": "_gen_billing_period",
        # Dữ liệu giấy phép lái xe
        "license_class": "_gen_license_class",
        "license_conditions": "_gen_license_conditions",
        "vehicle_type": "_gen_vehicle_type",
    }

    def __init__(self, locale: str = "en_AU"):
        """
        Khởi tạo FakerFactory với locale được chỉ định.

        Tham số:
            locale: Mã locale (ví dụ: 'en_AU', 'en_US', 'vi_VN').
        """
        self.locale = locale
        self.faker = Faker(locale)
        # Faker phụ trợ cho tiếng Anh chuẩn
        self.faker_en = Faker("en_US")
        logger.debug("Khởi tạo FakerFactory với locale: %s", locale)

    def generate_value(
        self,
        faker_type: str,
        options: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Sinh một giá trị dựa trên loại trường được chỉ định.

        Tham số:
            faker_type: Tên kiểu dữ liệu cần sinh.
            options: Tùy chọn bổ sung cho trường này.
            context: Ngữ cảnh đã sinh (dùng để đảm bảo tính nhất quán).

        Trả về:
            Chuỗi giá trị đã sinh.
        """
        options = options or {}
        context = context or {}

        handler_name = self.FAKER_TYPE_HANDLERS.get(faker_type)
        if not handler_name:
            logger.warning("Kiểu faker không xác định: '%s', dùng giá trị mặc định.", faker_type)
            return self.faker.word()

        handler = getattr(self, handler_name, None)
        if not handler:
            logger.error("Không tìm thấy phương thức xử lý: %s", handler_name)
            return ""

        try:
            return handler(options=options, context=context)
        except Exception as loi:
            logger.error("Lỗi khi sinh giá trị cho kiểu '%s': %s", faker_type, loi)
            return ""

    def generate_document_fields(
        self,
        fields_config: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Sinh toàn bộ các trường dữ liệu cho một tài liệu.

        Tham số:
            fields_config: Danh sách cấu hình các trường từ base.json.

        Trả về:
            Từ điển ánh xạ key -> giá trị đã sinh.
        """
        result: Dict[str, str] = {}

        for field_cfg in fields_config:
            key = field_cfg.get("key", "")
            if not key:
                continue

            faker_type = field_cfg.get("faker_type", "")
            options = field_cfg.get("options", {})

            value = self.generate_value(faker_type, options=options, context=result)

            # Áp dụng chuyển đổi chữ hoa nếu cấu hình yêu cầu
            if field_cfg.get("uppercase", False):
                value = value.upper()

            # Áp dụng chuyển đổi chữ thường nếu cần
            if field_cfg.get("lowercase", False):
                value = value.lower()

            result[key] = value

        logger.debug("Đã sinh %d trường dữ liệu.", len(result))
        return result

    # ---------------------------------------------------------------
    # Các phương thức sinh dữ liệu cá nhân
    # ---------------------------------------------------------------

    def _gen_full_name(self, options: Dict, context: Dict) -> str:
        return self.faker.name()

    def _gen_first_name(self, options: Dict, context: Dict) -> str:
        gender = context.get("gender", "").lower()
        if gender == "male":
            return self.faker.first_name_male()
        elif gender == "female":
            return self.faker.first_name_female()
        return self.faker.first_name()

    def _gen_last_name(self, options: Dict, context: Dict) -> str:
        return self.faker.last_name()

    def _gen_gender(self, options: Dict, context: Dict) -> str:
        choices = options.get("choices", ["Male", "Female"])
        return random.choice(choices)

    def _gen_date_of_birth(self, options: Dict, context: Dict) -> str:
        min_age = options.get("min_age", 18)
        max_age = options.get("max_age", 75)
        dob = self.faker.date_of_birth(minimum_age=min_age, maximum_age=max_age)
        fmt = options.get("format", "%d/%m/%Y")
        return dob.strftime(fmt)

    def _gen_nationality(self, options: Dict, context: Dict) -> str:
        nationalities = [
            "Australian", "American", "British", "Canadian",
            "New Zealander", "German", "French", "Japanese",
            "Korean", "Chinese", "Indian", "Brazilian",
        ]
        return random.choice(nationalities)

    def _gen_country(self, options: Dict, context: Dict) -> str:
        return self.faker.country()

    def _gen_country_code(self, options: Dict, context: Dict) -> str:
        codes = ["AUS", "USA", "GBR", "CAN", "NZL", "DEU", "FRA", "JPN"]
        return random.choice(codes)

    # ---------------------------------------------------------------
    # Các phương thức sinh số tài liệu
    # ---------------------------------------------------------------

    def _gen_passport_number(self, options: Dict, context: Dict) -> str:
        prefix = random.choice(["P", "A", "E", "N"])
        digits = "".join(random.choices(string.digits, k=7))
        return f"{prefix}{digits}"

    def _gen_id_number(self, options: Dict, context: Dict) -> str:
        length = options.get("length", 9)
        return "".join(random.choices(string.digits, k=length))

    def _gen_license_number(self, options: Dict, context: Dict) -> str:
        letters = "".join(random.choices(string.ascii_uppercase, k=2))
        digits = "".join(random.choices(string.digits, k=6))
        return f"{letters}{digits}"

    def _gen_medicare_number(self, options: Dict, context: Dict) -> str:
        # Định dạng Medicare Úc: XXXX XXXXX X
        group1 = "".join(random.choices(string.digits, k=4))
        group2 = "".join(random.choices(string.digits, k=5))
        group3 = random.choice(string.digits)
        return f"{group1} {group2} {group3}"

    def _gen_account_number(self, options: Dict, context: Dict) -> str:
        length = options.get("length", 10)
        return "".join(random.choices(string.digits, k=length))

    # ---------------------------------------------------------------
    # Các phương thức sinh ngày tháng
    # ---------------------------------------------------------------

    def _gen_issue_date(self, options: Dict, context: Dict) -> str:
        years_ago = options.get("years_ago", random.randint(1, 5))
        issue = date.today() - timedelta(days=years_ago * 365)
        fmt = options.get("format", "%d/%m/%Y")
        return issue.strftime(fmt)

    def _gen_expiry_date(self, options: Dict, context: Dict) -> str:
        years_ahead = options.get("years_ahead", random.randint(1, 10))
        expiry = date.today() + timedelta(days=years_ahead * 365)
        fmt = options.get("format", "%d/%m/%Y")
        return expiry.strftime(fmt)

    def _gen_date_formatted(self, options: Dict, context: Dict) -> str:
        start = options.get("start_year", 2000)
        end = options.get("end_year", 2024)
        rand_date = self.faker.date_between(
            start_date=date(start, 1, 1),
            end_date=date(end, 12, 31),
        )
        fmt = options.get("format", "%d/%m/%Y")
        return rand_date.strftime(fmt)

    # ---------------------------------------------------------------
    # Các phương thức sinh địa chỉ và liên lạc
    # ---------------------------------------------------------------

    def _gen_address(self, options: Dict, context: Dict) -> str:
        return self.faker.address().replace("\n", ", ")

    def _gen_street_address(self, options: Dict, context: Dict) -> str:
        return self.faker.street_address()

    def _gen_city(self, options: Dict, context: Dict) -> str:
        return self.faker.city()

    def _gen_state(self, options: Dict, context: Dict) -> str:
        try:
            return self.faker.state()
        except AttributeError:
            states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
            return random.choice(states)

    def _gen_postcode(self, options: Dict, context: Dict) -> str:
        try:
            return self.faker.postcode()
        except AttributeError:
            return f"{random.randint(1000, 9999)}"

    def _gen_phone_number(self, options: Dict, context: Dict) -> str:
        return self.faker.phone_number()

    def _gen_email(self, options: Dict, context: Dict) -> str:
        return self.faker.email()

    def _gen_company(self, options: Dict, context: Dict) -> str:
        return self.faker.company()

    def _gen_company_abn(self, options: Dict, context: Dict) -> str:
        """Sinh số ABN (Australian Business Number) giả."""
        digits = [random.randint(0, 9) for _ in range(11)]
        return " ".join([
            "".join(map(str, digits[:2])),
            "".join(map(str, digits[2:5])),
            "".join(map(str, digits[5:8])),
            "".join(map(str, digits[8:11])),
        ])

    # ---------------------------------------------------------------
    # Sinh dữ liệu MRZ hộ chiếu
    # ---------------------------------------------------------------

    def _gen_mrz_line1(self, options: Dict, context: Dict) -> str:
        """
        Sinh dòng MRZ đầu tiên (Machine Readable Zone).
        Định dạng: P<COD<<HO_TEN<<<<<<<<<<<<<<<<<<<<<<<<
        """
        country = context.get("country_code", "AUS")[:3].upper()
        last_name = context.get("last_name", self.faker.last_name()).upper()
        first_name = context.get("first_name", self.faker.first_name()).upper()

        # Chuẩn hóa: loại bỏ ký tự không hợp lệ
        last_name = "".join(c if c.isalpha() else "<" for c in last_name)
        first_name = "".join(c if c.isalpha() else "<" for c in first_name)

        name_field = f"{last_name}<<{first_name}"
        name_field = name_field.ljust(39, "<")[:39]
        return f"P<{country}{name_field}"

    def _gen_mrz_line2(self, options: Dict, context: Dict) -> str:
        """
        Sinh dòng MRZ thứ hai.
        Định dạng: XXXXXXXXXYYYMMDDXZZZZZZZZZZXXXXXXXXXX
        """
        passport_no = context.get("passport_number", self._gen_passport_number({}, {}))
        passport_no = "".join(c if c.isalnum() else "0" for c in passport_no).ljust(9, "<")[:9]
        check1 = str(random.randint(0, 9))

        country = context.get("country_code", "AUS")[:3].upper()

        # Ngày sinh dạng YYMMDD
        dob_str = context.get("date_of_birth", "01/01/1990")
        try:
            dob = date(int(dob_str[-4:]), int(dob_str[3:5]), int(dob_str[:2]))
            dob_mrz = dob.strftime("%y%m%d")
        except (ValueError, IndexError):
            dob_mrz = "900101"
        check2 = str(random.randint(0, 9))

        gender = context.get("gender", random.choice(["M", "F"]))[0].upper()
        if gender not in ("M", "F"):
            gender = "M"

        # Ngày hết hạn
        expiry_mrz = (date.today() + timedelta(days=3650)).strftime("%y%m%d")
        check3 = str(random.randint(0, 9))

        personal = "".join(random.choices(string.digits + "<", k=14))
        check4 = str(random.randint(0, 9))
        check5 = str(random.randint(0, 9))

        return f"{passport_no}{check1}{country}{dob_mrz}{check2}{gender}{expiry_mrz}{check3}{personal}{check4}{check5}"

    # ---------------------------------------------------------------
    # Sinh dữ liệu hóa đơn tiện ích
    # ---------------------------------------------------------------

    def _gen_utility_account_number(self, options: Dict, context: Dict) -> str:
        prefix = random.choice(["GAS", "ELC", "WTR", "NET"])
        digits = "".join(random.choices(string.digits, k=10))
        return f"{prefix}-{digits}"

    def _gen_meter_reading(self, options: Dict, context: Dict) -> str:
        reading = random.randint(1000, 99999)
        return f"{reading:,}"

    def _gen_amount_due(self, options: Dict, context: Dict) -> str:
        amount = round(random.uniform(50, 500), 2)
        return f"${amount:,.2f}"

    def _gen_billing_period(self, options: Dict, context: Dict) -> str:
        start = date.today() - timedelta(days=90)
        end = date.today() - timedelta(days=1)
        return f"{start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')}"

    # ---------------------------------------------------------------
    # Sinh dữ liệu bằng lái xe
    # ---------------------------------------------------------------

    def _gen_license_class(self, options: Dict, context: Dict) -> str:
        classes = ["C", "R", "LR", "MR", "HR", "HC", "MC"]
        return random.choice(classes)

    def _gen_license_conditions(self, options: Dict, context: Dict) -> str:
        conditions = ["NONE", "01", "02", "C", "B"]
        return random.choice(conditions)

    def _gen_vehicle_type(self, options: Dict, context: Dict) -> str:
        types = ["Car", "Motorcycle", "Truck", "Bus", "Van"]
        return random.choice(types)
