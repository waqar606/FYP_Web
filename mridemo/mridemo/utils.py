from rest_framework.response import Response


def custom_response(response_status, data, message, success=True):
    Response.status_code = response_status
    msg = {"success": success, "data": data, "message": message}
    return Response(msg)
