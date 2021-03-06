"""Small client implementation to use the oep API.

Version: 0.1

Example usage: create a table, insert data, retrieve data, delete table

cli = OEPClient(token='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', host='its10098.its.kfa-juelich.de')

table = 'my_awesome_test_table'
definition = {
    "columns": [
        {"name": "id", "data_type": "bigint", "is_nullable": "NO"},
        {"name": "field1", "data_type": "varchar(128)", "is_nullable": "NO"},
        {"name": "field2", "data_type": "integer", "is_nullable": "YES"}
    ],
    "constraints": [{"constraint_type": "PRIMARY KEY", "constraint_parameter": "id"}]
}
data =  [
    {'id': 1, 'field1': 'test', 'field2': 100},
    {'id': 2, 'field1': 'test2', 'field2': None}
]

cli.create_table(table, definition)
cli.insert_table(table, data)
return_data = cli.select_table(table)
cli.drop_table(table)

"""

import logging

import requests


class OEPClient:
    """Small client implementation to use the oep API."""

    def __init__(
        self,
        token=None,
        protocol="https",
        host="openenergy-platform.org",
        api_version="v0",
        default_schema="model_draft",
        **_kwargs
    ):
        """
        Args:
            token(str): your API token
            protocol(str, optional): should be https, unless you are in a local test environment
            host(str, optional): host of the oep platform. default is "openenergy-platform.org"
            api_version(str, optional): currently only "v0"
            default_schema(str, optional): the default schema for the tables, usually "model_draft"
        """
        self.headers = {"Authorization": "Token %s" % token} if token else {}
        self.api_url = "%s://%s/api/%s/" % (protocol, host, api_version)
        self.default_schema = default_schema

    def _get_table_url(self, table, schema=None):
        """Return base api url for table.

        Args:
            table(str): table name. Must be valid postgres table name,
                all lowercase, only letters, numbers and underscore
            schema(str, optional): table schema name.
                defaults to self.default_schema which is usually "model_draft"
        """
        schema = schema or self.default_schema
        return self.api_url + "schema/%s/tables/%s/" % (schema, table)

    def _request(self, method, url, expected_status, jsondata=None):
        """Send a request and perform basic check for results

        Args:
            method(str): http method, that will be passed on to `requests.request`
            url(str): request url
            expected_status(int): expected http status code.
                if result has a different code, an error will be raised
            jsondata(object, optional): payload that will be send as json in the request.
        Returns:
            result object from returned json data
        """
        res = requests.request(
            url=url, method=method, json=jsondata, headers=self.headers
        )
        logging.info("%d %s %s", res.status_code, method, url)
        res_json = res.json()
        if res.status_code != expected_status:
            raise Exception(res_json)
        return res_json

    def create_table(self, table, definition, schema=None):
        """Create table.

        Args:
            table(str): table name. Must be valid postgres table name,
                all lowercase, only letters, numbers and underscore
            definition(object): column and constraint definitions
                according to the oep specifications

                Notes:
                * data_type should be understood by postgresql database
                * is_nullable: "YES" or "NO"
                * the first column should be the primary key
                  and a numeric column with the name `id` for full functionality om the platform

                Example:
                {
                    "columns": [
                        {"name": "id", "data_type": "bigint", "is_nullable": "NO"},
                        {"name": "field1", "data_type": "varchar(128)", "is_nullable": "NO"},
                        {"name": "field2", "data_type": "integer", "is_nullable": "YES"}
                    ],
                    "constraints": [
                        {"constraint_type": "PRIMARY KEY", "constraint_parameter": "id"}
                    ]
                }

            schema(str, optional): table schema name.
                defaults to self.default_schema which is usually "model_draft"
        """
        url = self._get_table_url(table=table, schema=schema)
        return self._request("PUT", url, 201, {"query": definition})

    def drop_table(self, table, schema=None):
        """Drop table.

        Args:
            table(str): table name. Must be valid postgres table name,
                all lowercase, only letters, numbers and underscore
            schema(str, optional): table schema name.
                defaults to self.default_schema which is usually "model_draft"
        """
        url = self._get_table_url(table=table, schema=schema)
        return self._request("DELETE", url, 200)

    def select_table(self, table, schema=None):
        """Select all rows from table.

        Args:
            table(str): table name. Must be valid postgres table name,
                all lowercase, only letters, numbers and underscore
            schema(str, optional): table schema name.
                defaults to self.default_schema which is usually "model_draft"

        Returns:
            list of records(dict: column_name -> value)
        """
        url = self._get_table_url(table=table, schema=schema) + "rows/"
        res = self._request("GET", url, 200)
        logging.info("returned %d records", len(res))
        return res

    def insert_table(self, table, data, schema=None):
        """Insert records into table.

        Args:
            table(str): table name. Must be valid postgres table name,
                all lowercase, only letters, numbers and underscore
            data(list): list of records(dict: column_name -> value)
            schema(str, optional): table schema name.
                defaults to self.default_schema which is usually "model_draft"
        """
        url = self._get_table_url(table=table, schema=schema) + "rows/new"
        res = self._request("POST", url, 201, {"query": data})
        logging.info("inserted %d records", len(data))
        return res
